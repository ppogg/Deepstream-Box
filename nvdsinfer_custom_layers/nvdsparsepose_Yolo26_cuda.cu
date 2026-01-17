/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <nvdscommon.h>
#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
 

static uint16_t layer_offsets[3]       = {0};
static const uint8_t YOLO26_STRIDES[3] = {8, 16, 32};
static float *h_pinned_kpts_buffer     = nullptr;
static bool is_buffer_initialized      = false;

static thrust::device_vector<NvDsInferInstanceMaskInfo> objects_26k(DECODEFREESIZE);
static thrust::device_vector<int> count_26k(1, 0);
static thrust::device_vector<float> kpts_26k(DECODEFREESIZE * KPT_NUM_POSE * 3);

 
 
static __global__ void decodeYOLO26Tensor_cuda(NvDsInferInstanceMaskInfo *binfo, float *input, uint8_t grid_h,
                                               uint8_t grid_w, uint8_t stride, uint16_t netW, uint16_t netH,
                                               __half threshold, uint16_t box_offset, int *global_count,
                                               float *d_kpts_buffer)
{
    uint16_t total_grid = grid_h * grid_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (idx >= total_grid)
        return;
 
    __half confidence = __float2half(input[idx]);
    if (confidence < threshold)
    {
        return;
    }
 
    uint8_t row      = idx / grid_w;
    uint8_t col      = idx % grid_w;
    uint8_t KPT_NUMS = KPT_NUM_POSE * 3;
    __half human_kpts[KPT_NUM_POSE * 3];
 
    for (uint8_t i = 0; i <= KPT_NUMS; i += 3)
    {
        __half kpt_x = __float2half(input[idx + total_grid * (i + 1)]);
        __half kpt_y = __float2half(input[idx + total_grid * (i + 2)]);
        __half conf  = 1.0;
 
        human_kpts[i]     = (kpt_x + __float2half(col) + _HALF_0_5) * __float2half(stride);
        human_kpts[i + 1] = (kpt_y + __float2half(row) + _HALF_0_5) * __float2half(stride);
        human_kpts[i + 2] = conf;
    }
 
    __half x0 = find_min(human_kpts, KPT_NUMS, 0);
    __half y0 = find_min(human_kpts, KPT_NUMS, 1);
    __half x1 = find_max(human_kpts, KPT_NUMS, 0);
    __half y1 = find_max(human_kpts, KPT_NUMS, 1);
 
    uint8_t write_pos = atomicAdd(global_count, 1);
 
    if (write_pos >= DECODEFREESIZE) return;
 
    NvDsInferInstanceMaskInfo *obj = &binfo[write_pos];
    obj[0].detectionConfidence = __half2float(confidence);
    obj[0].left    = __half2float(x0) - 5;
    obj[0].top     = __half2float(y0) - 5;
    obj[0].width   = __half2float(x1 - x0) + 5;
    obj[0].height  = __half2float(y1 - y0) + 5;
    obj[0].classId = 0;
 
    uint32_t kpt_offset     = write_pos * KPT_NUMS;
    float *current_kpts_ptr = d_kpts_buffer + kpt_offset;
 
    for (uint8_t p = 0; p < KPT_NUMS; p++)
    {
        current_kpts_ptr[p] = __half2float(human_kpts[p]);
    }
}
 
static bool NvDsInferParseYOLO26_cuda_parallel(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                               NvDsInferNetworkInfo const &networkInfo,
                                               NvDsInferParseDetectionParams const &detectionParams,
                                               std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
    if (!is_buffer_initialized)
    {
        cudaHostAlloc(&h_pinned_kpts_buffer, 
                      DECODEFREESIZE * KPT_NUM_POSE * 3 * sizeof(float), 
                      cudaHostAllocDefault);

        is_buffer_initialized = true;
    }
 
    if (outputLayersInfo.size() < KPT_NUM_LAYER)
    {
        std::cerr << "Expected 3 output layers for YOLO26, got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }
 
    for (uint8_t i = 1; i < KPT_NUM_LAYER; ++i)
    {
        layer_offsets[i] = layer_offsets[i - 1] +
                           outputLayersInfo[i - 1].inferDims.d[1] * 
                           outputLayersInfo[i - 1].inferDims.d[2];
    }
 
    thrust::fill(count_26k.begin(), count_26k.end(), 0);
    cudaDeviceSynchronize();
 
    cudaStream_t streams[KPT_NUM_LAYER];
    for (uint8_t i = 0; i < KPT_NUM_LAYER; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }
 
    for (uint8_t layer_idx = 0; layer_idx < KPT_NUM_LAYER; ++layer_idx)
    {
        const NvDsInferLayerInfo &layer = outputLayersInfo[layer_idx];
        uint8_t grid_h      = layer.inferDims.d[1];
        uint8_t grid_w      = layer.inferDims.d[2];
        uint8_t stride      = YOLO26_STRIDES[layer_idx];
        uint16_t total_grid = grid_h * grid_w;
 
        int threads_per_block = 128;
        int blocks_per_grid   = (total_grid + threads_per_block - 1) / threads_per_block;
        float *data = (float *)layer.buffer;
 
        decodeYOLO26Tensor_cuda<<<blocks_per_grid, threads_per_block, 0, streams[layer_idx]>>>(
            thrust::raw_pointer_cast(objects_26k.data()),
            data,
            grid_h,
            grid_w,
            stride,
            networkInfo.width,
            networkInfo.height,
            Threshold,
            layer_offsets[layer_idx],
            thrust::raw_pointer_cast(count_26k.data()),
            thrust::raw_pointer_cast(kpts_26k.data()));
    }
 
    for (uint8_t i = 0; i < KPT_NUM_LAYER; ++i)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
 
    uint8_t valid_count;
    cudaMemcpy(&valid_count,
               thrust::raw_pointer_cast(count_26k.data()),
               sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
 
    if (valid_count > DECODEFREESIZE) valid_count = DECODEFREESIZE;
    
    objectList.resize(valid_count);
    thrust::copy(objects_26k.begin(), objects_26k.begin() + valid_count, objectList.begin());
 
    cudaMemcpy(h_pinned_kpts_buffer,
               thrust::raw_pointer_cast(kpts_26k.data()),
               valid_count * KPT_NUM_POSE * 3 * sizeof(float),
               cudaMemcpyDeviceToHost);
 
    for (int i = 0; i < valid_count; ++i) {
        objectList[i].mask_width  = networkInfo.width;
        objectList[i].mask_height = networkInfo.height;
        objectList[i].mask_size   = sizeof(float) * KPT_NUM_POSE * 3;

        objectList[i].mask = new float[KPT_NUM_POSE * 3];
        memcpy(objectList[i].mask, &h_pinned_kpts_buffer[i * KPT_NUM_POSE * 3], objectList[i].mask_size);
    }
 
    return true;
}
 
extern "C" bool NvDsInferParseYolo26Pose_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
    nvtxRangePush("NvDsInferParseYOLO26");
 
    bool ret = NvDsInferParseYOLO26_cuda_parallel(
        outputLayersInfo, networkInfo, detectionParams, objectList);
 
    nvtxRangePop();
    return ret;
}
 
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26Pose_cuda);