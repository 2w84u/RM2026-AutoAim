#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>

#ifdef SPVISION_WITH_OPENVINO
#include <openvino/openvino.hpp>
#endif

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  struct PendingFrame
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;
    cv::Mat dnn_output;
#ifdef SPVISION_WITH_OPENVINO
    cv::Mat ov_input;  // keeps the OpenVINO input buffer alive while async inference runs
    ov::InferRequest infer_request;
#endif
  };

  bool use_openvino_ = false;
  cv::dnn::Net net_;
#ifdef SPVISION_WITH_OPENVINO
  ov::Core core_;
  ov::CompiledModel compiled_model_;
#endif
  std::string device_;
  YOLO yolo_;

  tools::ThreadSafeQueue<PendingFrame> queue_{
    16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP