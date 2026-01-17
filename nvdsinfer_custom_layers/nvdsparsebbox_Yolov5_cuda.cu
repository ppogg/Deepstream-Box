/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


static uint16_t layer_offsets[3]        = {0};
static const uint8_t YOLOv5_STRIDES[3]  = {8, 16, 32};
static __half *DEVICE_YOLOv5_ANCHORS[3] = {nullptr, nullptr, nullptr};
static bool anchors_initialized         = false;

thrust::device_vector<NvDsInferParseObjectInfo> objects_v5b(ANCHORBASESIZE);

static const __half HOST_YOLOv5_ANCHORS[3][6] = {
    {10.0, 13.0, 16.0, 30.0, 33.0, 23.0},
    {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},
    {116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};

extern "C" bool NvDsInferParseCustomYolov5_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);

__device__ __host__ inline __half sigmoid(float x)
{
    float result = 1.0f / (1.0f + expf(-x));
    return __float2half(result);
}

static bool initDeviceAnchors()
{
    if (anchors_initialized)
    {
        return true;
    }

    for (uint8_t i = 0; i < DET_NUM_LAYER; i++)
    {
        cudaError_t err = cudaMalloc(
            &DEVICE_YOLOv5_ANCHORS[i],
            6 * sizeof(__half));

        if (err != cudaSuccess)
        {
            std::cerr << "Failed to allocate device memory for anchors: "
                      << cudaGetErrorString(err) << std::endl;
            for (uint8_t j = 0; j < i; j++)
            {
                if (DEVICE_YOLOv5_ANCHORS[j] != nullptr)
                {
                    cudaFree(DEVICE_YOLOv5_ANCHORS[j]);
                    DEVICE_YOLOv5_ANCHORS[j] = nullptr;
                }
            }
            return false;
        }
        err = cudaMemcpy(DEVICE_YOLOv5_ANCHORS[i],
                         HOST_YOLOv5_ANCHORS[i],
                         6 * sizeof(__half),
                         cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            std::cerr << "Failed to copy anchors to device: "
                      << cudaGetErrorString(err) << std::endl;
            cudaFree(DEVICE_YOLOv5_ANCHORS[i]);
            DEVICE_YOLOv5_ANCHORS[i] = nullptr;
            return false;
        }
    }

    anchors_initialized = true;
    return true;
}

__global__ void decodeYOLOv5Tensor_cuda(NvDsInferParseObjectInfo *binfo,
                                        float *input, uint8_t grid_h, uint8_t grid_w,
                                        uint8_t stride, const __half *anchors,
                                        uint16_t netW, uint16_t netH, __half threshold,
                                        uint16_t box_offset, uint8_t num_classes)
{
    uint16_t total_grid = grid_h * grid_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_grid)
        return;

    uint8_t info_len_i = 5 + num_classes;
    uint8_t row        = idx / grid_w;
    uint8_t col        = idx % grid_w;

    for (uint8_t k = 0; k < 3; k++)
    {
        int output_idx    = box_offset + idx * 3 + k;
        int anchor_offset = k * info_len_i * total_grid;
        __half box_prob   = sigmoid(input[anchor_offset + idx + 4 * total_grid]);

        if (box_prob < threshold)
        {
            binfo[output_idx].detectionConfidence = 0.0;
            continue;
        }

        uint8_t class_id = 0;
        __half cls_prob = 0.0;

        for (uint8_t i = 5; i < info_len_i; ++i)
        {
            __half p = sigmoid(input[anchor_offset + idx + i * total_grid]);
            if (p > cls_prob)
            {
                cls_prob = p;
                class_id = i - 5;
            }
        }

        __half confidence = box_prob * cls_prob;

        // YOLOv5解码公式
        __half px = sigmoid(input[anchor_offset + idx + 0 * total_grid]);
        __half py = sigmoid(input[anchor_offset + idx + 1 * total_grid]);
        __half pw = sigmoid(input[anchor_offset + idx + 2 * total_grid]);
        __half ph = sigmoid(input[anchor_offset + idx + 3 * total_grid]);

        __half bx = (px * _HALF_2 - _HALF_0_5 + __float2half(col)) * __float2half(stride);
        __half by = (py * _HALF_2 - _HALF_0_5 + __float2half(row)) * __float2half(stride);
        __half bw = pw * pw * _HALF_4 * anchors[2 * k];
        __half bh = ph * ph * _HALF_4 * anchors[2 * k + 1];

        __half x0 = bx - bw / _HALF_2;
        __half y0 = by - bh / _HALF_2;
        __half x1 = x0 + bw;
        __half y1 = y0 + bh;

        x0 = fmaxf(0.0f, fminf(__float2half(netW), x0));
        y0 = fmaxf(0.0f, fminf(__float2half(netH), y0));
        x1 = fmaxf(0.0f, fminf(__float2half(netW), x1));
        y1 = fmaxf(0.0f, fminf(__float2half(netH), y1));

        binfo[output_idx].left     = __half2float(x0);
        binfo[output_idx].top      = __half2float(y0);
        binfo[output_idx].width    = __half2float(x1 - x0);
        binfo[output_idx].height   = __half2float(y1 - y0);
        binfo[output_idx].detectionConfidence = confidence;
        binfo[output_idx].classId  = class_id;
    }
}

static bool NvDsInferParseYOLOv5_cuda_parallel(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                               NvDsInferNetworkInfo const &networkInfo,
                                               NvDsInferParseDetectionParams const &detectionParams,
                                               std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.size() < DET_NUM_LAYER)
    {
        std::cerr << "Expected 3 output layers for YOLOv5, got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    if (!initDeviceAnchors())
    {
        std::cerr << "Failed to initialize device anchors" << std::endl;
        return false;
    }

    cudaStream_t streams[DET_NUM_LAYER];
    for (uint8_t i = 0; i < DET_NUM_LAYER; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    for (uint8_t i = 1; i < DET_NUM_LAYER; ++i)
    {
        layer_offsets[i] = layer_offsets[i - 1] +
                           outputLayersInfo[i - 1].inferDims.d[1] 
                           * outputLayersInfo[i - 1].inferDims.d[2];
    }

    for (uint8_t layer_idx = 0; layer_idx < 3; ++layer_idx)
    {
        const NvDsInferLayerInfo &layer = outputLayersInfo[layer_idx];
        uint8_t grid_h      = layer.inferDims.d[1];
        uint8_t grid_w      = layer.inferDims.d[2];
        uint8_t stride      = YOLOv5_STRIDES[layer_idx];
        uint16_t total_grid = grid_h * grid_w;

        int threads_per_block = 256;
        int blocks_per_grid = (total_grid + threads_per_block - 1) / threads_per_block;
        float *data = (float *)layer.buffer;

        decodeYOLOv5Tensor_cuda<<<blocks_per_grid, threads_per_block, 0, streams[layer_idx]>>>(
            thrust::raw_pointer_cast(objects_v5b.data()), 
            data, 
            grid_h, 
            grid_w, 
            stride,
            DEVICE_YOLOv5_ANCHORS[layer_idx], 
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

    objectList.resize(ANCHORBASESIZE);
    thrust::copy(objects_v5b.begin(), objects_v5b.end(), objectList.begin());

    return true;
}

extern "C" bool NvDsInferParseCustomYolov5_cuda(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    // NVTX性能分析范围
    nvtxRangePush("NvDsInferParseYOLOv5");

    bool ret = NvDsInferParseYOLOv5_cuda_parallel(
        outputLayersInfo, networkInfo, detectionParams, objectList);

    nvtxRangePop();
    return ret;
}

/* 验证自定义函数定义正确 */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolov5_cuda);