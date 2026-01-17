#include <cassert>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <glib.h>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseSwimTransformer(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                      NvDsInferNetworkInfo const &networkInfo,
                                      NvDsInferParseDetectionParams const &detectionParams,
                                      std::vector<NvDsInferParseObjectInfo> &objectList);

static bool NvDsInferParseCustomSwimTransformer(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                        NvDsInferNetworkInfo const &networkInfo,
                                        NvDsInferParseDetectionParams const &detectionParams,
                                        std::vector<NvDsInferParseObjectInfo> &objectList)
{
  if (outputLayersInfo.empty())
  {
    std::cerr << "ERROR - Could not find output layer" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo &output = outputLayersInfo[0];
  const float *preBuffer           = (float *)(output.buffer);
  size_t channelsSize              = output.inferDims.d[0];

  g_print("[DEBUG] channelsSize: %ld, classId: %d, confidence: %f\n", 
          channelsSize, (uint8_t)preBuffer[0], preBuffer[1]);
          
  objectList.resize(1);
  objectList[0].left    = 10.0f;
  objectList[0].top     = 10.0f;
  objectList[0].width   = networkInfo.width - 20;
  objectList[0].height  = networkInfo.height - 20;
  objectList[0].detectionConfidence = preBuffer[1];
  objectList[0].classId = (uint8_t)preBuffer[0];

  return true;
}

extern "C" bool
NvDsInferParseSwimTransformer(std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
                      NvDsInferParseDetectionParams const &detectionParams, std::vector<NvDsInferParseObjectInfo> &objectList)
{
  return NvDsInferParseCustomSwimTransformer(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseSwimTransformer);
