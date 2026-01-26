#include <cuda_fp16.h>
#include <nvdscommon.h>
#include <cmath>
#include <iostream>

static const __half HOST_v5Face_ANCHORS[3][6] = {{4.0, 5.0, 8.0, 10.0, 13.0, 16.0},
                                                 {23.0, 29.0, 43.0, 55.0, 73.0, 105.0},
                                                 {146.0, 217.0, 231.0, 300.0, 335.0, 433.0}};

static uint16_t layer_offsets[3] = {0};
static const uint8_t v5Face_STRIDES[3] = {8, 16, 32};
static __half* DEVICE_v5Face_ANCHORS[3] = {nullptr, nullptr, nullptr};
static bool anchors_initialized = false;
static float* h_pinned_kpts_buffer = nullptr;
static bool is_buffer_initialized = false;

static thrust::device_vector<NvDsInferInstanceMaskInfo> objects_v5face(DECODEFREESIZE);
static thrust::device_vector<int> count_v5face(1, 0);
static thrust::device_vector<float> kpts_v5face(DECODEFREESIZE * KPT_NUM_FACE * 2);

extern "C" bool NvDsInferParseCustomv5Face_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                                NvDsInferNetworkInfo const& networkInfo,
                                                NvDsInferParseDetectionParams const& detectionParams,
                                                std::vector<NvDsInferInstanceMaskInfo>& objectList);

__device__ __host__ inline __half sigmoid(float x) {
    float result = 1.0f / (1.0f + expf(-x));
    return __float2half(result);
}

static bool initDeviceAnchors() {
    if (anchors_initialized) {
        return true;
    }

    for (uint8_t i = 0; i < KPT_NUM_LAYER; i++) {
        cudaError_t err = cudaMalloc(&DEVICE_v5Face_ANCHORS[i], 6 * sizeof(__half));

        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for anchors: " << cudaGetErrorString(err) << std::endl;
            for (uint8_t j = 0; j < i; j++) {
                if (DEVICE_v5Face_ANCHORS[j] != nullptr) {
                    cudaFree(DEVICE_v5Face_ANCHORS[j]);
                    DEVICE_v5Face_ANCHORS[j] = nullptr;
                }
            }
            return false;
        }
        err = cudaMemcpy(DEVICE_v5Face_ANCHORS[i], HOST_v5Face_ANCHORS[i], 6 * sizeof(__half), cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            std::cerr << "Failed to copy anchors to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(DEVICE_v5Face_ANCHORS[i]);
            DEVICE_v5Face_ANCHORS[i] = nullptr;
            return false;
        }
    }

    anchors_initialized = true;
    return true;
}

static std::vector<NvDsInferInstanceMaskInfo> nonMaximumSuppression(std::vector<NvDsInferInstanceMaskInfo> binfo) {
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min) {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](NvDsInferInstanceMaskInfo& bbox1, NvDsInferInstanceMaskInfo& bbox2) -> float {
        float overlapX = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
        float overlapY = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
        float area1 = (bbox1.width) * (bbox1.height);
        float area2 = (bbox2.width) * (bbox2.height);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const NvDsInferInstanceMaskInfo& b1, const NvDsInferInstanceMaskInfo& b2) {
                         return b1.detectionConfidence > b2.detectionConfidence;
                     });

    std::vector<NvDsInferInstanceMaskInfo> out;
    for (auto i : binfo) {
        bool keep = true;
        for (auto j : out) {
            if (keep) {
                float overlap = computeIoU(i, j);
                keep = overlap <= __half2float(nmsThreshold);
            } else {
                break;
            }
        }
        if (keep) {
            out.push_back(i);
        }
    }
    return out;
}

static __global__ void decodev5FaceTensor_cuda(NvDsInferInstanceMaskInfo* binfo, float* input, uint8_t grid_h, uint8_t grid_w,
                                        uint8_t stride, const __half* anchors, uint16_t netW, uint16_t netH,
                                        __half threshold, uint16_t box_offset, int* global_count,
                                        float* d_kpts_buffer) 
{
    uint16_t total_grid = grid_h * grid_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_grid)
        return;

    uint8_t info_len_i = 16;
    uint8_t row = idx / grid_w;
    uint8_t col = idx % grid_w;
    uint8_t KPT_NUMS = KPT_NUM_FACE * 2;
    __half face_kpts[KPT_NUM_FACE * 2];

    for (uint8_t k = 0; k < KPT_NUM_LAYER; k++) {
        int anchor_offset = k * info_len_i * total_grid;
        int output_idx = box_offset + idx * KPT_NUM_LAYER + k;
        __half box_prob = sigmoid(input[anchor_offset + idx + 4 * total_grid]);

        if (box_prob < threshold) {
            binfo[output_idx].detectionConfidence = 0.0;
            continue;
        }

        __half cls_prob = input[anchor_offset + idx + 15 * total_grid];
        __half confidence = box_prob * cls_prob;

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

        for (uint8_t i = 0; i <= KPT_NUMS; i += 2) {
            __half kpt_x = sigmoid(input[idx + anchor_offset + (5 + i) * total_grid]);
            __half kpt_y = sigmoid(input[idx + anchor_offset + (6 + i) * total_grid]);

            face_kpts[i] = (kpt_x * _HALF_2 + __float2half(col)) * __float2half(stride);
            face_kpts[i + 1] = (kpt_y * _HALF_2 + __float2half(row)) * __float2half(stride);
        }

        uint8_t write_pos = atomicAdd(global_count, 1);
        if (write_pos >= DECODEFREESIZE)
            return;

        NvDsInferInstanceMaskInfo* obj = &binfo[write_pos];
        obj[0].detectionConfidence = __half2float(confidence);
        obj[0].left = __half2float(x0);
        obj[0].top = __half2float(y0);
        obj[0].width = __half2float(x1 - x0);
        obj[0].height = __half2float(y1 - y0);
        obj[0].classId = 0;

        uint32_t kpt_offset = write_pos * KPT_NUMS;
        float* current_kpts_ptr = d_kpts_buffer + kpt_offset;

        for (uint8_t p = 0; p < KPT_NUMS; p++) {
            current_kpts_ptr[p] = __half2float(face_kpts[p]);
        }
    }
}

static bool NvDsInferParsev5Face_cuda_parallel(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                               NvDsInferNetworkInfo const& networkInfo,
                                               NvDsInferParseDetectionParams const& detectionParams,
                                               std::vector<NvDsInferInstanceMaskInfo>& objectList) 
{
    if (!is_buffer_initialized)
    {
        cudaHostAlloc(&h_pinned_kpts_buffer, 
                      DECODEFREESIZE * KPT_NUM_POSE * 2 * sizeof(float), 
                      cudaHostAllocDefault);

        is_buffer_initialized = true;
    }

    if (outputLayersInfo.size() < KPT_NUM_LAYER) {
        std::cerr << "Expected 3 output layers for v5Face, got " << outputLayersInfo.size() << std::endl;
        return false;
    }

    if (!initDeviceAnchors()) {
        std::cerr << "Failed to initialize device anchors" << std::endl;
        return false;
    }

    cudaStream_t streams[KPT_NUM_LAYER];
    for (uint8_t i = 0; i < KPT_NUM_LAYER; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (uint8_t i = 1; i < KPT_NUM_LAYER; ++i) {
        layer_offsets[i] =
                layer_offsets[i - 1] + outputLayersInfo[i - 1].inferDims.d[1] * outputLayersInfo[i - 1].inferDims.d[2];
    }

    float threshold = detectionParams.perClassThreshold[0];

    for (uint8_t layer_idx = 0; layer_idx < KPT_NUM_LAYER; ++layer_idx) {
        const NvDsInferLayerInfo& layer = outputLayersInfo[layer_idx];
        uint8_t grid_h = layer.inferDims.d[1];
        uint8_t grid_w = layer.inferDims.d[2];
        uint8_t stride = v5Face_STRIDES[layer_idx];
        uint16_t total_grid = grid_h * grid_w;

        int threads_per_block = 256;
        int blocks_per_grid = (total_grid + threads_per_block - 1) / threads_per_block;
        float* data = (float*)layer.buffer;

        decodev5FaceTensor_cuda<<<blocks_per_grid, threads_per_block, 0, streams[layer_idx]>>>(
                thrust::raw_pointer_cast(objects_v5face.data()), data, grid_h, grid_w, stride,
                DEVICE_v5Face_ANCHORS[layer_idx], networkInfo.width, networkInfo.height, Threshold,
                layer_offsets[layer_idx], thrust::raw_pointer_cast(count_v5face.data()),
                thrust::raw_pointer_cast(kpts_v5face.data()));
    }

    for (uint8_t i = 0; i < KPT_NUM_LAYER; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    uint8_t valid_count;
    cudaMemcpy(&valid_count, thrust::raw_pointer_cast(count_v5face.data()), sizeof(uint8_t), cudaMemcpyDeviceToHost);

    if (valid_count > DECODEFREESIZE)
        valid_count = DECODEFREESIZE;

    std::vector<NvDsInferInstanceMaskInfo> hostList(valid_count);
    thrust::copy(objects_v5face.begin(), objects_v5face.begin() + valid_count, hostList.begin());

    cudaMemcpy(h_pinned_kpts_buffer, thrust::raw_pointer_cast(kpts_v5face.data()),
               valid_count * KPT_NUM_POSE * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < valid_count; ++i) {
        hostList[i].mask_width = networkInfo.width;
        hostList[i].mask_height = networkInfo.height;
        hostList[i].mask_size = sizeof(float) * KPT_NUM_POSE * 2;

        hostList[i].mask = new float[KPT_NUM_POSE * 2];
        memcpy(hostList[i].mask, &h_pinned_kpts_buffer[i * KPT_NUM_POSE * 2], hostList[i].mask_size);
    }

    objectList = nonMaximumSuppression(hostList);

    return true;
}

extern "C" bool NvDsInferParseCustomv5Face_cuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                                NvDsInferNetworkInfo const& networkInfo,
                                                NvDsInferParseDetectionParams const& detectionParams,
                                                std::vector<NvDsInferInstanceMaskInfo>& objectList) {
    nvtxRangePush("NvDsInferParsev5Face");

    bool ret = NvDsInferParsev5Face_cuda_parallel(outputLayersInfo, networkInfo, detectionParams, objectList);

    nvtxRangePop();
    return ret;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomv5Face_cuda);