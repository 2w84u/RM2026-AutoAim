#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <string>

#ifdef SPVISION_WITH_OPENVINO
#include <openvino/openvino.hpp>
#endif

#include "armor.hpp"

namespace auto_aim
{
class Classifier
{
public:
  explicit Classifier(const std::string & config_path);

  void classify(Armor & armor);

  void ovclassify(Armor & armor);
  std::string inference_graph();

private:
  cv::dnn::Net net_;
  bool opencv_forwarded_ = false;
#ifdef SPVISION_WITH_OPENVINO
  ov::Core core_;
  ov::CompiledModel compiled_model_;
#endif
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP
