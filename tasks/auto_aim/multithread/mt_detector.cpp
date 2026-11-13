#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

#include "tools/infer_backend.hpp"

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

  if (tools::is_onnx_model(model_path)) {
    // ONNX -> OpenCV DNN (CUDA when device: GPU and ENABLE_CUDA_DNN is on)
    net_ = cv::dnn::readNetFromONNX(model_path);
    tools::configure_dnn_backend(net_, device_);
  } else {
    use_openvino_ = true;
#ifdef SPVISION_WITH_OPENVINO
    auto model = core_.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();

    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, 640, 640, 3})  // TODO
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);

    input.model().set_layout("NCHW");

    input.preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
      .scale(255.0);

    model = ppp.build();
    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
#else
    throw std::runtime_error(
      "OpenVINO backend is disabled at build time; provide an .onnx model for CUDA DNN (got: " +
      model_path + ")");
#endif
  }

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // preproces
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});

  PendingFrame frame;
  frame.img = img.clone();
  frame.t = t;

  if (use_openvino_) {
#ifdef SPVISION_WITH_OPENVINO
    frame.ov_input = input;
    ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, frame.ov_input.data);
    frame.infer_request = compiled_model_.create_infer_request();
    frame.infer_request.set_input_tensor(input_tensor);
    frame.infer_request.start_async();
#endif
  } else {
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);
    net_.setInput(blob);
    cv::Mat out = net_.forward();
    frame.dnn_output = out.reshape(1, out.size[1]);
  }

  queue_.push(std::move(frame));
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto frame = queue_.pop();

  cv::Mat output;
  if (use_openvino_) {
#ifdef SPVISION_WITH_OPENVINO
    frame.infer_request.wait();
    auto output_tensor = frame.infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    output = cv::Mat(output_shape[1], output_shape[2], CV_32F, output_tensor.data()).clone();
#endif
  } else {
    output = frame.dnn_output;
  }

  auto x_scale = static_cast<double>(640) / frame.img.rows;
  auto y_scale = static_cast<double>(640) / frame.img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, frame.img, 0);  //暂不支持ROI

  return {std::move(armors), frame.t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto frame = queue_.pop();

  cv::Mat output;
  if (use_openvino_) {
#ifdef SPVISION_WITH_OPENVINO
    frame.infer_request.wait();
    auto output_tensor = frame.infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    output = cv::Mat(output_shape[1], output_shape[2], CV_32F, output_tensor.data()).clone();
#endif
  } else {
    output = frame.dnn_output;
  }

  auto x_scale = static_cast<double>(640) / frame.img.rows;
  auto y_scale = static_cast<double>(640) / frame.img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, frame.img, 0);  //暂不支持ROI

  return {frame.img, std::move(armors), frame.t};
}

}  // namespace multithread

}  // namespace auto_aim
