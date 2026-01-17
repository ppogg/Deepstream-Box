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
#ifndef NVDSCOMMON_H
#define NVDSCOMMON_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>      
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "nvtx3/nvToolsExt.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>

#define DET_NUM_LAYER 3
#define KPT_NUM_LAYER 3
#define SEG_NUM_LAYER 3

#define KPT_NUM_FACE 5
#define KPT_NUM_POSE 17

#define DECODEFREESIZE 255
#define ANCHORBASESIZE 25200
#define ANCHORFREESIZE 8400

#define _HALF_2   __float2half(2.0f)
#define _HALF_4   __float2half(4.0f)
#define _HALF_8   __float2half(8.0f)
#define _HALF_0_5 __float2half(0.5f)

static const __half Threshold    = 0.25;
static const __half nmsThreshold = 0.5;
static const uint8_t NUM_CLASSES = 80;

typedef struct {
    float left;
    float top;
    float width;
    float height;
    float conf;
    int classId;
    float mask[KPT_NUM_POSE * 3]; 
} KptsInfo;

struct isValidElement
{
    __host__ __device__
    bool operator()(const KptsInfo &info)
    {
        return info.conf > 0.25; 
    }
};

__device__ __host__ inline __half find_min(__half *array, uint8_t length, bool y)
{
    __half min = 640;
    if (y) {
        for (uint8_t i = 1; i < length; i += 3) {
            if (array[i] < min) {
                min = array[i];
            }
        }
    }
    else {
        for (uint8_t i = 0; i < length; i += 3) {
            if (array[i] < min) {
                min = array[i];
            }
        }
    }
    return min;
}

__device__ __host__ inline __half find_max(__half *array, uint8_t length, bool y)
{
    __half max = 0;
    if (y) {
        for (uint8_t i = 1; i < length; i += 3) {
            if (array[i] > max) {
                max = array[i];
            }
        }
    }
    else
    {
        for (uint8_t i = 0; i < length; i += 3) {
            if (array[i] > max) {
                max = array[i];
            }
        }
    }
    return max;
}

#endif