#include <cassert>
#include <algorithm>
#include <iostream>
#include <cstring>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseFastsamSeg(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                      NvDsInferNetworkInfo const &networkInfo,
                                      NvDsInferParseDetectionParams const &detectionParams,
                                      std::vector<NvDsInferInstanceMaskInfo> &objectList);

static float clamp(float val, float minVal, float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}

static std::vector<NvDsInferInstanceMaskInfo> decodeTensorFastsamSeg(const float *output, size_t outputSize,
                                                                  size_t channelsSize, uint netW, uint netH,
                                                                  const std::vector<float> &preclusterThreshold)
{
  std::vector<NvDsInferInstanceMaskInfo> objects;
  for (size_t n = 0; n < outputSize; ++n)
  {
    float maxProb = output[n * channelsSize + 4];
    int maxIndex = (int)output[n * channelsSize + 5];

    if (maxProb < preclusterThreshold[maxIndex])
    {
      continue;
    }

    float x1 = output[n * channelsSize + 0];
    float y1 = output[n * channelsSize + 1];
    float x2 = output[n * channelsSize + 2];
    float y2 = output[n * channelsSize + 3];

    NvDsInferInstanceMaskInfo b;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left    = x1;
    b.width   = clamp(x2 - x1, 0, netW);
    b.top     = y1;
    b.height  = clamp(y2 - y1, 0, netH);
    b.classId = maxIndex + n;
    b.detectionConfidence = maxProb;

    size_t maskSize = channelsSize - 6;
    b.mask          = new float[maskSize];
    b.mask_width    = netW / 8;
    b.mask_height   = netH / 8;
    b.mask_size     = sizeof(float) * maskSize;
    std::memcpy(b.mask, output + n * channelsSize + 6, sizeof(float) * maskSize);

    objects.push_back(b);
  }

  return objects;
}

static bool
NvDsInferParseCustomFastsamSeg(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                            NvDsInferNetworkInfo const &networkInfo,
                            NvDsInferParseDetectionParams const &detectionParams,
                            std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
  if (outputLayersInfo.empty())
  {
    std::cerr << "ERROR - Could not find output layer" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo &output = outputLayersInfo[0];
  size_t outputSize   = output.inferDims.d[0];
  size_t channelsSize = output.inferDims.d[1];

  std::vector<NvDsInferInstanceMaskInfo> objects = decodeTensorFastsamSeg((const float *)(output.buffer), outputSize,
                                                                       channelsSize, networkInfo.width, networkInfo.height,
                                                                       detectionParams.perClassPreclusterThreshold);

  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseFastsamSeg(std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
                      NvDsInferParseDetectionParams const &detectionParams, std::vector<NvDsInferInstanceMaskInfo> &objectList)
{
  return NvDsInferParseCustomFastsamSeg(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseFastsamSeg);
