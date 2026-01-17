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


static uint16_t layer_offsets[3]        = {0};
static const uint8_t Master_DFL_DIM     = 16;
static const uint8_t Master_STRIDES[3]  = {8, 16, 32};

thrust::device_vector<NvDsInferParseObjectInfo> objects_master(ANCHORFREESIZE);

extern "C" bool NvDsInferParseCustomMaster_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);

__device__ __host__ inline __half softmax_sum(float *arr, uint8_t length)
{
    float sum_exp = 0.0f;
    float dfl_sum = 0.0f;
    float max_val = arr[0];

    for (uint8_t i = 1; i < length; ++i)
    {
        if (arr[i] > max_val)
        {
            max_val = arr[i];
        }
    }

    for (uint8_t i = 0; i < length; ++i)
    {
        sum_exp += expf(arr[i] - max_val);
    }

    for (uint8_t i = 0; i < length; ++i)
    {
        float prob = expf(arr[i] - max_val) / sum_exp;
        dfl_sum += prob * i;
    }

    return __float2half(dfl_sum);
}

__global__ void decodeMasterTensor_cuda(NvDsInferParseObjectInfo *binfo, float *input, uint8_t grid_h,
                                        uint8_t grid_w, uint8_t stride, uint16_t netW, uint16_t netH,
                                        __half threshold, uint16_t box_offset, uint8_t num_classes)
{
    uint16_t total_grid = grid_h * grid_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_grid)
        return;

    int output_idx    = box_offset + idx;
    uint8_t row       = idx / grid_w;
    uint8_t col       = idx % grid_w;
    __half confidence = __float2half(input[idx]);

    if (confidence < threshold)
    {
        binfo[output_idx].detectionConfidence = 0.0;
        return;
    }

    uint8_t class_id = 0;
    __half cls_prob  = __float2half(0.0f);

    int cls_start_idx = 65 * total_grid; // 65 = 1(conf) + 64(dfl)

    for (uint8_t i = 0; i < num_classes; ++i)
    {
        int cls_idx    = cls_start_idx + i * total_grid + idx;
        __half cls_val = __float2half(input[cls_idx]);

        if (cls_val > cls_prob)
        {
            cls_prob = cls_val;
            class_id = i;
        }
    }

    // DFL解码框坐标
    float dflv[4][Master_DFL_DIM];
    for (int i = 0; i < Master_DFL_DIM; ++i)
    {
        dflv[0][i] = input[(1 + i + 0 * Master_DFL_DIM) * total_grid + idx];
        dflv[1][i] = input[(1 + i + 1 * Master_DFL_DIM) * total_grid + idx];
        dflv[2][i] = input[(1 + i + 2 * Master_DFL_DIM) * total_grid + idx];
        dflv[3][i] = input[(1 + i + 3 * Master_DFL_DIM) * total_grid + idx];
    }

    // Master解码公式
    __half reg_x0 = softmax_sum(&dflv[0][0], Master_DFL_DIM);
    __half reg_y0 = softmax_sum(&dflv[1][0], Master_DFL_DIM);
    __half reg_x1 = softmax_sum(&dflv[2][0], Master_DFL_DIM);
    __half reg_y1 = softmax_sum(&dflv[3][0], Master_DFL_DIM);

    __half x0 = (__float2half(col) + _HALF_0_5 - reg_x0) * __float2half(stride);
    __half y0 = (__float2half(row) + _HALF_0_5 - reg_y0) * __float2half(stride);
    __half x1 = (__float2half(col) + _HALF_0_5 + reg_x1) * __float2half(stride);
    __half y1 = (__float2half(row) + _HALF_0_5 + reg_y1) * __float2half(stride);

    x0 = fmaxf(0.0f, fminf(__float2half(netW), x0));
    y0 = fmaxf(0.0f, fminf(__float2half(netH), y0));
    x1 = fmaxf(0.0f, fminf(__float2half(netW), x1));
    y1 = fmaxf(0.0f, fminf(__float2half(netH), y1));

    binfo[output_idx].left    = __half2float(x0);
    binfo[output_idx].top     = __half2float(y0);
    binfo[output_idx].width   = __half2float(x1 - x0);
    binfo[output_idx].height  = __half2float(y1 - y0);
    binfo[output_idx].detectionConfidence = __half2float(cls_prob);
    binfo[output_idx].classId = class_id;
}

static bool NvDsInferParseMaster_cuda_parallel(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                               NvDsInferNetworkInfo const &networkInfo,
                                               NvDsInferParseDetectionParams const &detectionParams,
                                               std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.size() < DET_NUM_LAYER)
    {
        std::cerr << "Expected 3 output layers for Master, got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    for (uint8_t i = 1; i < DET_NUM_LAYER; ++i)
    {
        layer_offsets[i] = layer_offsets[i - 1] +
                           outputLayersInfo[i - 1].inferDims.d[1] 
                           * outputLayersInfo[i - 1].inferDims.d[2];
    }

    cudaStream_t streams[DET_NUM_LAYER];
    for (uint8_t i = 0; i < DET_NUM_LAYER; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    for (uint8_t layer_idx = 0; layer_idx < DET_NUM_LAYER; ++layer_idx)
    {
        const NvDsInferLayerInfo &layer = outputLayersInfo[layer_idx];
        uint8_t grid_h                  = layer.inferDims.d[1];
        uint8_t grid_w                  = layer.inferDims.d[2];
        uint8_t stride                  = Master_STRIDES[layer_idx];
        uint16_t total_grid             = grid_h * grid_w;

        int threads_per_block = 256;
        int blocks_per_grid = (total_grid + threads_per_block - 1) / threads_per_block;
        float *data = (float *)layer.buffer;

        decodeMasterTensor_cuda<<<blocks_per_grid, threads_per_block, 0, streams[layer_idx]>>>(
            thrust::raw_pointer_cast(objects_master.data()),
            data,
            grid_h,
            grid_w,
            stride,
            networkInfo.width,
            networkInfo.height,
            Threshold,
            layer_offsets[layer_idx],
            NUM_CLASSES);
    }

    for (uint8_t i = 0; i < DET_NUM_LAYER; ++i)
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

    objectList.resize(ANCHORFREESIZE);
    thrust::copy(objects_master.begin(), objects_master.end(), objectList.begin());

    return true;
}

extern "C" bool NvDsInferParseCustomMaster_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    nvtxRangePush("NvDsInferParseMaster");

    bool ret = NvDsInferParseMaster_cuda_parallel(
        outputLayersInfo, networkInfo, detectionParams, objectList);

    nvtxRangePop();
    return ret;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMaster_cuda);