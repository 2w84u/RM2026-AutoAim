#pragma once

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tasks/auto_aim/yolo.hpp"

namespace detector_verify
{
inline constexpr std::array<const char *, auto_aim::detect_stage_count> stage_keys = {
  "roi", "resize", "pack", "request", "bind", "forward", "output", "layout", "decode",
  "nms", "construct", "pattern", "classify", "refine", "filter", "debug_draw", "other"};

inline void write_file(const std::filesystem::path & path, const std::string & contents)
{
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out || !(out << contents)) throw std::runtime_error("无法写入: " + path.string());
}

inline nlohmann::json statistics(std::vector<double> values)
{
  if (values.empty()) return {{"samples", 0}};
  const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
  std::sort(values.begin(), values.end());
  auto percentile = [&](double p) {
    const auto index = static_cast<std::size_t>(std::ceil(p * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
  };
  return {{"samples", values.size()}, {"mean_ms", sum / values.size()},
    {"p50_ms", percentile(0.50)}, {"p95_ms", percentile(0.95)},
    {"min_ms", values.front()}, {"max_ms", values.back()}};
}

class Diagnostics
{
public:
  Diagnostics(bool enabled, int warmup, std::filesystem::path directory)
  : enabled_(enabled), warmup_(warmup), directory_(std::move(directory)) {}

  void start(const YAML::Node & yaml, const std::string & config_path)
  {
    if (!enabled_) return;
    // Set this before constructing YOLO: CUDA fallback messages appear on the first forward.
    if (cv::utils::logging::getLogLevel() < cv::utils::logging::LOG_LEVEL_INFO)
      cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
    std::filesystem::create_directories(directory_);
    const auto model_name = yaml["yolo_name"].as<std::string>();
    const auto model_path = yaml[model_name + "_model_path"].as<std::string>();
    metadata_ = {
      {"opencv_version", CV_VERSION}, {"opencv_threads", cv::getNumThreads()},
      {"config", std::filesystem::absolute(config_path).string()},
      {"model", model_name}, {"model_path", std::filesystem::absolute(model_path).string()},
      {"requested_device", yaml["device"].as<std::string>("CPU")},
      {"warmup_frames", warmup_}, {"control_transport", "not constructed"},
      {"timing_note", "Host wall time. forward includes device transfers, synchronization "
        "and any CPU fallback; these are not individual CUDA kernel timings."},
      {"placement_note", "DOT nodes may contain fused layers. OCV means no backend node "
        "for that group, not a measured CPU time or fallback percentage. Input/shape "
        "groups and expensive compute groups must be interpreted separately."}};
    if (std::filesystem::exists(model_path))
      metadata_["model_bytes"] = std::filesystem::file_size(model_path);
    if (yaml["class_num"]) metadata_["class_num"] = yaml["class_num"].as<int>();
    metadata_["use_roi"] = yaml["use_roi"].as<bool>(false);
    metadata_["use_traditional_config"] = yaml["use_traditional"].as<bool>(false);
#ifdef NDEBUG
    metadata_["build"] = "NDEBUG (optimized build expected)";
#else
    metadata_["build"] = "debug / NDEBUG not defined";
#endif
#ifdef SPVISION_WITH_CUDA_DNN
    metadata_["cuda_dnn_compiled"] = true;
#else
    metadata_["cuda_dnn_compiled"] = false;
#endif
#ifdef SPVISION_WITH_OPENVINO
    metadata_["openvino_compiled"] = true;
#else
    metadata_["openvino_compiled"] = false;
#endif
    metadata_["cuda_devices"] = nlohmann::json::array();
    metadata_["cuda_dnn_targets"] = nlohmann::json::array();
    try {
      const int count = cv::cuda::getCudaEnabledDeviceCount();
      metadata_["cuda_device_count"] = count;
      for (int i = 0; i < count; ++i) {
        cv::cuda::DeviceInfo device(i);
        metadata_["cuda_devices"].push_back({{"id", i}, {"name", device.name()},
          {"compute_capability", fmt::format("{}.{}", device.majorVersion(), device.minorVersion())},
          {"compatible", device.isCompatible()}});
      }
      for (const auto target : cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA))
        metadata_["cuda_dnn_targets"].push_back(target == cv::dnn::DNN_TARGET_CUDA_FP16
          ? "CUDA_FP16" : target == cv::dnn::DNN_TARGET_CUDA ? "CUDA_FP32" : "other");
    } catch (const cv::Exception & e) {
      metadata_["cuda_query_error"] = e.what();
    }
    write_file(directory_ / "opencv-build.txt", cv::getBuildInformation());
    fmt::print("[diagnose] OpenCV {} | {} | CUDA compiled={} devices={} targets={}\n",
      CV_VERSION, metadata_["build"].get<std::string>(),
      metadata_["cuda_dnn_compiled"].get<bool>(),
      metadata_.value("cuda_device_count", 0), metadata_["cuda_dnn_targets"].dump());
    fmt::print("[diagnose] 排除前 {} 帧预热；默认无窗口；forward 为含传输/同步/回退的墙钟耗时。\n",
      warmup_);
    std::fflush(stdout);
  }

  nlohmann::json sample(
    int frame_number, const cv::Mat & input, double total_ms,
    const auto_aim::DetectionTiming & timing)
  {
    if (!enabled_) return nullptr;
    if (frame_number == 1) {
      metadata_["image_shape_hwc"] = {input.rows, input.cols, input.channels()};
      metadata_["first_frame_ms"] = total_ms;
    }
    nlohmann::json result = {{"valid", timing.valid},
      {"warmup", frame_number <= warmup_}};
    if (frame_number > warmup_) detect_ms_.push_back(total_ms);
    if (!timing.valid) return result;
    result["total_ms"] = timing.total_ms;
    result["preprocessing_ms"] = timing.preprocessing_ms();
    result["postprocessing_ms"] = timing.postprocessing_ms();
    result["raw_candidates"] = timing.raw_candidates;
    result["score_candidates"] = timing.score_candidates;
    result["nms_candidates"] = timing.nms_candidates;
    result["armors"] = timing.armors;
    result["stages_ms"] = nlohmann::json::object();
    for (std::size_t i = 0; i < stage_keys.size(); ++i) {
      if (frame_number > warmup_) {
        stage_ms_[i].push_back(timing.stage_ms[i]);
        stage_measured_[i] = stage_measured_[i] || timing.measured[i];
      }
      if (!timing.measured[i]) continue;
      result["stages_ms"][stage_keys[i]] = timing.stage_ms[i];
    }
    return result;
  }

  nlohmann::json finish(auto_aim::YOLO & yolo, int frames)
  {
    if (!enabled_) return nullptr;
    metadata_["frames"] = frames;
    metadata_["detect"] = statistics(detect_ms_);
    metadata_["stages"] = nlohmann::json::object();
    fmt::print("\n[diagnose] 稳态统计：{} 帧（前 {} 帧不计）；单位 ms\n",
      detect_ms_.size(), std::min(frames, warmup_));
    fmt::print("  {:<34} {:>9} {:>9} {:>9} {:>9}\n", "stage", "mean", "p50", "p95", "max");
    auto print_stats = [](const char * name, const nlohmann::json & stats) {
      if (stats["samples"].get<std::size_t>() == 0) return;
      fmt::print("  {:<34} {:9.3f} {:9.3f} {:9.3f} {:9.3f}\n", name,
        stats["mean_ms"].get<double>(), stats["p50_ms"].get<double>(),
        stats["p95_ms"].get<double>(), stats["max_ms"].get<double>());
    };
    print_stats("Detect total", metadata_["detect"]);
    for (std::size_t i = 0; i < stage_keys.size(); ++i) {
      if (!stage_measured_[i] || stage_ms_[i].empty()) continue;
      auto stats = statistics(stage_ms_[i]);
      metadata_["stages"][stage_keys[i]] = stats;
      print_stats(auto_aim::detect_stage_names[i], stats);
    }
    metadata_["networks"] = nlohmann::json::array();
    if (frames > 0) {
      try {
        for (const auto & graph : yolo.inference_graphs()) {
          nlohmann::json network = {{"name", graph.name}, {"note", graph.note},
            {"cuda_groups", 0}, {"opencv_groups", 0},
            {"opencv_nodes", nlohmann::json::array()}};
          fmt::print("\n[diagnose] {}: {}\n", graph.name, graph.note);
          if (!graph.dot.empty()) {
            const auto path = directory_ / (graph.name + ".dot");
            write_file(path, graph.dot);
            network["dot_path"] = path.string();
            std::istringstream lines(graph.dot);
            std::string line;
            int cuda_groups = 0, opencv_groups = 0;
            while (std::getline(lines, line)) {
              const auto label = line.find("[label=\"");
              if (label == std::string::npos) continue;
              if (line.find("CUDA/", label) != std::string::npos) ++cuda_groups;
              const auto ocv = line.find("OCV/", label);
              if (ocv == std::string::npos) continue;
              ++opencv_groups;
              const auto start = label + std::string("[label=\"").size();
              const auto end = line.find("\" ", start);
              auto description = line.substr(start, end - start);
              for (std::size_t pos = 0; (pos = description.find("\\n", pos)) != std::string::npos;)
                description.replace(pos, 2, " | ");
              network["opencv_nodes"].push_back(description);
              fmt::print("  OpenCV group: {}\n", description);
            }
            network["cuda_groups"] = cuda_groups;
            network["opencv_groups"] = opencv_groups;
            fmt::print("  CUDA groups={} OpenCV groups={} (含融合组；不是回退耗时占比)\n",
              cuda_groups, opencv_groups);
            fmt::print("  graph -> {}\n", path.string());
          } else {
            network["graph_available"] = false;
          }
          metadata_["networks"].push_back(std::move(network));
        }
      } catch (const std::exception & e) {
        metadata_["graph_error"] = e.what();
        fmt::print("[diagnose] 导出执行图失败: {}\n", e.what());
      }
    }
    return metadata_;
  }

  void save(const nlohmann::json & report) const
  {
    if (!enabled_) return;
    const auto path = directory_ / "report.json";
    write_file(path, report.dump(1));
    fmt::print("[diagnose] report -> {}\n", path.string());
  }

private:
  bool enabled_;
  int warmup_;
  std::filesystem::path directory_;
  nlohmann::json metadata_;
  std::vector<double> detect_ms_;
  std::array<std::vector<double>, auto_aim::detect_stage_count> stage_ms_;
  std::array<bool, auto_aim::detect_stage_count> stage_measured_{};
};
}  // namespace detector_verify
