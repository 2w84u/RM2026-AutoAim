#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "calibration/chessboard.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明 }"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/img_with_q        | 输入图片目录 }"
  "{display        | false                    | 逐张显示角点并等待按键，默认批处理 }"
  "{output o       |                          | 标定结果 YAML 输出路径 }";

namespace fs = std::filesystem;

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, float distance)
{
  std::vector<cv::Point3f> points;
  for (int row = 0; row < pattern_size.height; ++row)
    for (int col = 0; col < pattern_size.width; ++col)
      points.emplace_back(col * distance, row * distance, 0);
  return points;
}

void load(
  const std::string & input_folder, const std::string & config_path, cv::Size & img_size,
  std::vector<std::vector<cv::Point3f>> & obj_points,
  std::vector<std::vector<cv::Point2f>> & img_points, bool display)
{
  const auto folder = fs::absolute(input_folder).lexically_normal();
  if (!fs::is_directory(folder))
    throw std::runtime_error(fmt::format("输入图片目录不存在：{}", folder.string()));

  const auto yaml = YAML::LoadFile(config_path);
  const cv::Size pattern_size(yaml["pattern_cols"].as<int>(), yaml["pattern_rows"].as<int>());
  const auto distance = yaml["center_distance_mm"].as<double>();
  if (pattern_size.width < 2 || pattern_size.height < 2 || !std::isfinite(distance) || distance <= 0)
    throw std::runtime_error("标定板内角点数量或间距无效");

  std::vector<fs::path> images;
  for (const auto & entry : fs::directory_iterator(folder)) {
    if (!entry.is_regular_file()) continue;
    auto extension = entry.path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp")
      images.push_back(entry.path());
  }
  std::sort(images.begin(), images.end());
  if (images.empty())
    throw std::runtime_error(fmt::format("目录内没有 JPG/JPEG/PNG/BMP 图片：{}", folder.string()));

  const auto object_points = centers_3d(pattern_size, static_cast<float>(distance));
  fmt::print("图片目录：{}\n共 {} 张，棋盘格 {}×{} 内角点，间距 {} mm\n",
    folder.string(), images.size(), pattern_size.width, pattern_size.height, distance);
  std::fflush(stdout);

  for (size_t i = 0; i < images.size(); ++i) {
    const auto & path = images[i];
    fmt::print("[{}/{}] 正在读取和识别 {} ...\n", i + 1, images.size(), path.filename().string());
    std::fflush(stdout);
    auto image = cv::imread(path.string());
    if (image.empty()) {
      fmt::print("[跳过] 图片无法解码：{}\n", path.string());
      continue;
    }
    if (img_size.empty()) img_size = image.size();
    if (image.size() != img_size)
      throw std::runtime_error(fmt::format(
        "图片分辨率不一致：{} 是 {}×{}，前面图片是 {}×{}；请分目录标定",
        path.filename().string(), image.cols, image.rows, img_size.width, img_size.height));

    std::vector<cv::Point2f> corners;
    const bool detected = calibration::find_chessboard_corners(image, pattern_size, corners);
    fmt::print("[{}] {}\n", detected ? "success" : "failure", path.filename().string());
    std::fflush(stdout);
    if (display) {
      cv::drawChessboardCorners(image, pattern_size, corners, detected);
      cv::resize(image, image, {}, 0.5, 0.5);
      cv::imshow("Any key: next; q/Esc: cancel", image);
      const int key = cv::waitKey(0);
      if (key == 'q' || key == 'Q' || key == 27)
        throw std::runtime_error("用户取消标定");
    }
    if (!detected) continue;
    img_points.push_back(std::move(corners));
    obj_points.push_back(object_points);
  }
  if (display) cv::destroyAllWindows();
  fmt::print("读图结束：{} / {} 张有效，分辨率 {}×{}\n",
    img_points.size(), images.size(), img_size.width, img_size.height);
  std::fflush(stdout);
  if (img_points.size() < 3)
    throw std::runtime_error("有效棋盘格照片不足 3 张，无法求解内参；请补拍完整棋盘格的不同倾斜姿态");
}

void print_yaml(
  const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, const cv::Size & image_size,
  size_t views, double rms, double mean_error, const std::string & output_path)
{
  YAML::Emitter result;
  result.SetDoublePrecision(17);
  const std::vector<double> camera_matrix_data(camera_matrix.begin<double>(), camera_matrix.end<double>());
  const std::vector<double> distort_coeffs_data(distort_coeffs.begin<double>(), distort_coeffs.end<double>());
  result << YAML::BeginMap;
  result << YAML::Comment(fmt::format("平均重投影误差: {:.6f}px; RMS: {:.6f}px", mean_error, rms));
  result << YAML::Key << "image_width" << YAML::Value << image_size.width;
  result << YAML::Key << "image_height" << YAML::Value << image_size.height;
  result << YAML::Key << "calibration_images" << YAML::Value << views;
  result << YAML::Key << "rms_reprojection_error" << YAML::Value << rms;
  result << YAML::Key << "mean_reprojection_error" << YAML::Value << mean_error;
  result << YAML::Key << "camera_matrix" << YAML::Value << YAML::Flow << camera_matrix_data;
  result << YAML::Key << "distort_coeffs" << YAML::Value << YAML::Flow << distort_coeffs_data;
  result << YAML::EndMap;
  fmt::print("\n{}\n", result.c_str());
  if (!output_path.empty()) {
    const fs::path path(output_path);
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path);
    file << result.c_str() << '\n';
    file.close();
    fmt::print("标定结果已写入：{}\n", fs::absolute(path).string());
  }
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto input_folder = cli.get<std::string>(0);
  const auto config_path = cli.get<std::string>("config-path");
  const auto display = cli.get<bool>("display");
  const auto output_path = cli.get<std::string>("output");
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }

  try {
    cv::Size image_size;
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    load(input_folder, config_path, image_size, object_points, image_points, display);

    fmt::print("开始内参求解（{} 张实拍照片，QR 分解，固定 k3）...\n", image_points.size());
    std::fflush(stdout);
    const auto start = std::chrono::steady_clock::now();
    cv::Mat camera_matrix, distort_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    const double rms = cv::calibrateCamera(
      object_points, image_points, image_size, camera_matrix, distort_coeffs, rvecs, tvecs,
      cv::CALIB_FIX_K3 | cv::CALIB_USE_QR,
      {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-10});
    if (!std::isfinite(rms) || !cv::checkRange(camera_matrix) || !cv::checkRange(distort_coeffs))
      throw std::runtime_error("标定求解没有得到有限参数，请补拍更清晰且倾斜角度不同的棋盘格照片");

    double error_sum = 0;
    size_t total_points = 0;
    for (size_t i = 0; i < object_points.size(); ++i) {
      std::vector<cv::Point2f> reprojected;
      cv::projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected);
      for (size_t j = 0; j < reprojected.size(); ++j)
        error_sum += cv::norm(image_points[i][j] - reprojected[j]);
      total_points += reprojected.size();
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fmt::print("求解完成，用时 {:.1f}s；RMS {:.6f}px，平均重投影误差 {:.6f}px\n",
      elapsed, rms, error_sum / total_points);
    print_yaml(camera_matrix, distort_coeffs, image_size, image_points.size(), rms,
      error_sum / total_points, output_path);
  } catch (const std::exception & e) {
    fmt::print(stderr, "标定失败：{}\n", e.what());
    return 1;
  }
  return 0;
}
