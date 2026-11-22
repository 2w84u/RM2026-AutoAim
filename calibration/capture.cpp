#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>

#include "calibration/chessboard.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path   | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o | assets/img_with_q        | 输出文件夹路径   }"
  "{images-only     | false                    | 仅保存图片，不连接云台 }";

namespace
{
namespace fs = std::filesystem;

void save_sample(
  const fs::path & folder, int index, const cv::Mat & image,
  const std::optional<Eigen::Quaterniond> & q)
{
  const auto image_path = folder / fmt::format("{}.jpg", index);
  const auto q_path = folder / fmt::format("{}.txt", index);
  if (fs::exists(image_path) || fs::exists(q_path))
    throw std::runtime_error("图片或姿态文件已存在，未覆盖");
  if (q && (!q->coeffs().allFinite() || std::abs(q->norm() - 1.0) > 1e-6))
    throw std::runtime_error("四元数无效，未保存");

  try {
    // Write pose first so a newly visible image always has its matching pose.
    if (q) {
      std::ofstream file;
      file.exceptions(std::ios::failbit | std::ios::badbit);
      file.open(q_path);
      file << std::setprecision(17) << q->w() << ' ' << q->x() << ' ' << q->y() << ' ' << q->z()
           << '\n';
      file.close();
    }
    if (!cv::imwrite(image_path.string(), image))
      throw std::runtime_error("图片写入失败");
  } catch (...) {
    // Remove files created by this attempt if either half failed to write.
    std::error_code ignored;
    fs::remove(image_path, ignored);
    if (q) fs::remove(q_path, ignored);
    throw;
  }
}

void capture_loop(const std::string & config_path, const fs::path & output_folder, bool images_only)
{
  const auto yaml = YAML::LoadFile(config_path);
  const cv::Size pattern_size(yaml["pattern_cols"].as<int>(), yaml["pattern_rows"].as<int>());
  tools::logger()->info("棋盘格内角点：{} 列 × {} 行", pattern_size.width, pattern_size.height);
  tools::logger()->info("实时预览；按 s 时检测棋盘格并保存原图，按 q 退出");

  tools::Exiter exiter;
  std::unique_ptr<io::Gimbal> gimbal;
  if (!images_only) {
    tools::logger()->info("同步记录 Gimbal yaw/pitch 合成姿态（roll=0），四元数顺序 w x y z");
    gimbal = std::make_unique<io::Gimbal>(config_path);
  }
  io::Camera camera(config_path);
  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  int index = 1;

  while (!exiter.exit()) {
    camera.read(image, timestamp);
    if (image.empty()) continue;

    // Associate pose with this image before detection/display adds any latency.
    // Missing/stale telemetry remains empty instead of becoming identity pose.
    const auto q = gimbal ? gimbal->try_q(timestamp) : std::nullopt;
    cv::Mat drawing;
    cv::resize(image, drawing, {}, 0.5, 0.5, cv::INTER_AREA);
    if (q) {
      const Eigen::Vector3d ypr = tools::eulers(*q, 2, 1, 0) * 180.0 / CV_PI;
      tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {20, 25}, {0, 0, 255}, 0.7, 1);
      tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {20, 50}, {0, 0, 255}, 0.7, 1);
      tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {20, 75}, {0, 0, 255}, 0.7, 1);
    } else {
      tools::draw_text(
        drawing, images_only ? "Images only" : "Waiting for matching gimbal pose",
        {20, 25}, {0, 0, 255}, 0.7, 1);
    }

    cv::imshow("Press s to detect/save, q to quit", drawing);
    const auto key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') break;
    if (key != 's' && key != 'S') continue;

    if (!images_only && !q) {
      tools::logger()->warn("没有与图像时刻匹配的有效姿态（允许偏差 50 ms），本帧未保存");
      continue;
    }

    // Full-resolution detection runs only on the frame selected for saving.
    // Keep its already matched pose while detection and disk writes take time.
    std::vector<cv::Point2f> corners;
    if (!calibration::find_chessboard_corners(image, pattern_size, corners)) {
      tools::logger()->warn("未检测到完整棋盘格，本帧未保存");
      continue;
    }

    // Reusing an output directory must not overwrite earlier captures.
    while (fs::exists(output_folder / fmt::format("{}.jpg", index)) ||
           fs::exists(output_folder / fmt::format("{}.txt", index)))
      ++index;
    try {
      save_sample(output_folder, index, image, q);
      tools::logger()->info(
        "[{}] 已保存{}：{}", index, images_only ? "图片" : "图片和四元数", output_folder.string());
      ++index;
    } catch (const std::exception & e) {
      tools::logger()->error("保存失败：{}", e.what());
    }
  }
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);
  const auto output_folder = cli.get<std::string>("output-folder");
  const auto images_only = cli.get<bool>("images-only");
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }

  try {
    std::filesystem::create_directories(output_folder);
    capture_loop(config_path, output_folder, images_only);
  } catch (const std::exception & e) {
    tools::logger()->error("采集失败：{}", e.what());
    return 1;
  }
  return 0;
}
