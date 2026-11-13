#ifndef TOOLS__INFER_BACKEND_HPP
#define TOOLS__INFER_BACKEND_HPP

#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>

#include <string>

#include "tools/logger.hpp"

namespace tools
{
// OpenCV DNN and TensorRT cannot load OpenVINO IR models. In this project the
// file extension is the single switch between the two backends:
//   *.onnx -> OpenCV DNN (CUDA when available)
//   *.xml  -> OpenVINO IR (only when the build has ENABLE_OPENVINO)
inline bool is_onnx_model(const std::string & path)
{
  return path.size() > 5 && path.compare(path.size() - 5, 5, ".onnx") == 0;
}

// device: "GPU"/"CUDA"/"GPU_FP16" -> DNN_BACKEND_CUDA + DNN_TARGET_CUDA_FP16,
// anything else -> CPU. Requires ENABLE_CUDA_DNN at configure time and an
// OpenCV build with the CUDA modules; otherwise it falls back to CPU.
inline void configure_dnn_backend(cv::dnn::Net & net, const std::string & device)
{
  const bool want_gpu = device == "GPU" || device == "CUDA" || device == "GPU_FP16";

#ifdef SPVISION_WITH_CUDA_DNN
  if (want_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    logger()->info("[infer] OpenCV DNN CUDA FP16 backend enabled");
    return;
  }
#endif

  if (want_gpu) {
    logger()->warn("[infer] device \"{}\" requested but CUDA DNN is unavailable, using CPU", device);
  }
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

}  // namespace tools

#endif  // TOOLS__INFER_BACKEND_HPP
