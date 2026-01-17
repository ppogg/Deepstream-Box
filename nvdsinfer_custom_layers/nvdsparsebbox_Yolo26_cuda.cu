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
static const uint8_t YOLO26_STRIDES[3]  = {8, 16, 32};

thrust::device_vector<NvDsInferParseObjectInfo> objects_26b(ANCHORFREESIZE);

extern "C" bool NvDsInferParseCustomYolo26_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);

__global__ void decodeYOLO26Tensor_cuda(NvDsInferParseObjectInfo *binfo, float *input, uint8_t grid_h,
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

    uint8_t class_id  = 0;
    __half cls_prob   = __float2half(0.0f);
    int cls_start_idx = 5 * total_grid;

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
    
    // YOLO26解码公式
    __half reg_x0 = __float2half(input[idx + 1 * total_grid]);
    __half reg_y0 = __float2half(input[idx + 2 * total_grid]);
    __half reg_x1 = __float2half(input[idx + 3 * total_grid]);
    __half reg_y1 = __float2half(input[idx + 4 * total_grid]);

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

static bool NvDsInferParseYOLO26_cuda_parallel(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                               NvDsInferNetworkInfo const &networkInfo,
                                               NvDsInferParseDetectionParams const &detectionParams,
                                               std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.size() < DET_NUM_LAYER)
    {
        std::cerr << "Expected 3 output layers for YOLO26, got "
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
        uint8_t grid_h      = layer.inferDims.d[1];
        uint8_t grid_w      = layer.inferDims.d[2];
        uint8_t stride      = YOLO26_STRIDES[layer_idx];
        uint16_t total_grid = grid_h * grid_w;

        int threads_per_block = 128;
        int blocks_per_grid = (total_grid + threads_per_block - 1) / threads_per_block;
        float *data = (float *)layer.buffer;

        decodeYOLO26Tensor_cuda<<<blocks_per_grid, threads_per_block, 0, streams[layer_idx]>>>(
            thrust::raw_pointer_cast(objects_26b.data()),
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

    // 将结果复制回主机
    objectList.resize(ANCHORFREESIZE);
    thrust::copy(objects_26b.begin(), objects_26b.end(), objectList.begin());

    return true;
}

extern "C" bool NvDsInferParseCustomYolo26_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    nvtxRangePush("NvDsInferParseYOLO26");

    bool ret = NvDsInferParseYOLO26_cuda_parallel(
        outputLayersInfo, networkInfo, detectionParams, objectList);

    nvtxRangePop();
    return ret;
}

/* 验证自定义函数定义正确 */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26_cuda);