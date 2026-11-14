#include <fmt/core.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <nlohmann/json.hpp>

#include "io/camera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"

namespace fs = std::filesystem;

struct Options
{
  std::string input;
  std::string config_path = "configs/jetson.yaml";
  std::string output_dir = "verify_out";
  std::string json_path;
  int max_frames = 0;
  int print_every = 0;
  bool save = true;
  bool show = false;
  bool quiet = false;
  bool camera = false;
  bool save_raw = false;
};

void print_usage()
{
  fmt::print(
    "用法: detector_verify [选项] <图片|图片目录|视频>\n"
    "\n"
    "  -h, --help                 显示本帮助\n"
    "  -c, --config-path PATH     yaml 配置（默认 configs/jetson.yaml）\n"
    "  -o, --output DIR           标注结果输出目录（默认 verify_out）\n"
    "  -m, --max-frames N         最多处理帧数，0=全部（默认 0）\n"
    "  -p, --print-every N        每 N 帧打印一次，0=自动（相机 30，文件 1）\n"
    "  -j, --json FILE            检测结果写入 JSON（用于 A/B 对比）\n"
    "      --save[=BOOL]          保存标注结果（默认 true）\n"
    "      --show                 实时窗口显示（需要桌面环境）\n"
    "  -q, --quiet                只打印汇总\n"
    "      --camera               直接读取相机（相机参数取自 config）\n"
    "      --save-raw             保存前 5 帧原始图像（诊断预处理）\n"
    "\n"
    "示例:\n"
    "  detector_verify --camera -c configs/jetson-ir.yaml -m 300 -p 30 --show\n"
    "  detector_verify match.avi -c configs/jetson-ir.yaml --json cand.json\n");
}

bool parse_bool(const std::string & v)
{
  return !(v == "false" || v == "0" || v == "no" || v == "off");
}

Options parse_args(int argc, char ** argv, bool & help)
{
  Options o;
  help = argc < 2;

  auto value = [&](int & i, const std::string & arg) -> std::string {
    const auto pos = arg.find('=');
    if (pos != std::string::npos) return arg.substr(pos + 1);
    if (i + 1 >= argc) throw std::runtime_error("缺少参数值: " + arg);
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const std::string key = arg.substr(0, arg.find('='));
    if (key == "-h" || key == "--help" || key == "--usage") {
      help = true;
    } else if (key == "-c" || key == "--config-path") {
      o.config_path = value(i, arg);
    } else if (key == "-o" || key == "--output") {
      o.output_dir = value(i, arg);
    } else if (key == "-m" || key == "--max-frames") {
      o.max_frames = std::stoi(value(i, arg));
    } else if (key == "-p" || key == "--print-every") {
      o.print_every = std::stoi(value(i, arg));
    } else if (key == "-j" || key == "--json") {
      o.json_path = value(i, arg);
    } else if (key == "--save") {
      o.save = arg.find('=') == std::string::npos ? true : parse_bool(value(i, arg));
    } else if (key == "--show") {
      o.show = true;
    } else if (key == "-q" || key == "--quiet") {
      o.quiet = true;
    } else if (key == "--camera") {
      o.camera = true;
    } else if (key == "--save-raw") {
      o.save_raw = true;
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::runtime_error("未知参数: " + arg);
    } else if (o.input.empty()) {
      o.input = arg;
    } else {
      throw std::runtime_error("多余的位置参数: " + arg);
    }
  }
  return o;
}

const std::vector<std::string> IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};

bool is_image(const fs::path & p)
{
  auto ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return std::find(IMAGE_EXT.begin(), IMAGE_EXT.end(), ext) != IMAGE_EXT.end();
}

std::string armor_name(const auto_aim::Armor & armor)
{
  const auto idx = static_cast<std::size_t>(armor.name);
  if (idx < auto_aim::ARMOR_NAMES.size()) return auto_aim::ARMOR_NAMES[idx];
  return "unknown";
}

void draw(cv::Mat & img, const std::list<auto_aim::Armor> & armors)
{
  for (const auto & armor : armors) {
    cv::rectangle(img, armor.box, {0, 255, 0}, 2);
    for (std::size_t i = 0; i < armor.points.size(); ++i) {
      cv::circle(img, armor.points[i], 3, {0, 0, 255}, -1);
      if (armor.points.size() == 4)
        cv::line(
          img, armor.points[i], armor.points[(i + 1) % armor.points.size()], {0, 255, 255}, 1);
    }
    const auto label = fmt::format("{} {:.2f}", armor_name(armor), armor.confidence);
    cv::putText(
      img, label, armor.box.tl() + cv::Point(0, -6), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
  }
}

nlohmann::json to_json(const std::list<auto_aim::Armor> & armors)
{
  nlohmann::json arr = nlohmann::json::array();
  for (const auto & armor : armors) {
    nlohmann::json det;
    det["name"] = armor_name(armor);
    det["confidence"] = armor.confidence;
    det["box"] = {armor.box.x, armor.box.y, armor.box.width, armor.box.height};
    nlohmann::json pts = nlohmann::json::array();
    for (const auto & p : armor.points) pts.push_back({p.x, p.y});
    det["points"] = pts;
    arr.push_back(det);
  }
  return arr;
}

int main(int argc, char * argv[])
{
  bool help = false;
  Options opt;
  try {
    opt = parse_args(argc, argv, help);
  } catch (const std::exception & e) {
    fmt::print("{}\n\n", e.what());
    print_usage();
    return 1;
  }
  if (help) {
    print_usage();
    return 0;
  }

  const auto & input = opt.input;
  const auto & config_path = opt.config_path;
  const auto & output_dir = opt.output_dir;
  const auto & json_path = opt.json_path;
  const auto max_frames = opt.max_frames;
  const bool save = opt.save;
  const bool show = opt.show;
  const bool quiet = opt.quiet;
  const bool use_camera = opt.camera;
  const bool save_raw = opt.save_raw;

  if (!use_camera && (input.empty() || !fs::exists(input))) {
    fmt::print("输入路径不存在: {}\n", input);
    return 1;
  }

  // 打印实际使用的模型，方便确认测的是哪一份
  {
    auto yaml = YAML::LoadFile(config_path);
    auto name = yaml["yolo_name"].as<std::string>();
    auto model = yaml[name + "_model_path"].as<std::string>();
    fmt::print("config : {}\n", config_path);
    fmt::print("model  : {} ({})\n", model, yaml["device"] ? yaml["device"].as<std::string>() : "CPU");
    fmt::print("class  : {}\n", yaml["class_num"] ? yaml["class_num"].as<int>() : -1);
  }

  auto_aim::YOLO yolo(config_path, false);  // debug=false：不弹窗、不保存中间图

  std::unique_ptr<io::Camera> camera;
  if (use_camera) {
    auto yaml = YAML::LoadFile(config_path);
    fmt::print(
      "camera : {} exposure={} ms\n",
      yaml["camera_name"] ? yaml["camera_name"].as<std::string>() : "?",
      yaml["exposure_ms"] ? yaml["exposure_ms"].as<std::string>() : "?");
    try {
      camera = std::make_unique<io::Camera>(config_path);
    } catch (const std::exception & e) {
      fmt::print("相机初始化失败: {}\n", e.what());
      return 1;
    }
  }

  if (save) fs::create_directories(output_dir);

  // 收集输入帧
  std::vector<cv::Mat> images;
  cv::VideoCapture video;
  std::vector<fs::path> image_paths;
  bool is_video = false;

  if (use_camera) {
    fmt::print("输入   : 相机（Ctrl+C 结束）\n");
  } else if (fs::is_directory(input)) {
    for (const auto & e : fs::directory_iterator(input))
      if (e.is_regular_file() && is_image(e.path())) image_paths.push_back(e.path());
    std::sort(image_paths.begin(), image_paths.end());
    fmt::print("输入   : 目录 {}（{} 张图片）\n", input, image_paths.size());
  } else if (is_image(input)) {
    image_paths.push_back(input);
    fmt::print("输入   : 图片 {}\n", input);
  } else {
    video.open(input);
    if (!video.isOpened()) {
      fmt::print("无法打开视频: {}\n", input);
      return 1;
    }
    is_video = true;
    fmt::print("输入   : 视频 {}（{} 帧 @ {:.1f} fps）\n", input,
               static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT)),
               video.get(cv::CAP_PROP_FPS));
  }

  std::fflush(stdout);  // 便于通过管道/远程终端实时查看

  cv::VideoWriter writer;
  const bool stream_mode = is_video || use_camera;
  nlohmann::json report;
  report["config"] = config_path;
  report["input"] = use_camera ? std::string("camera") : input;
  report["frames"] = nlohmann::json::array();

  int frames = 0, frames_with_det = 0, total_det = 0;
  double total_ms = 0;
  int bogus_conf = 0;  // 类别分数 > 1 通常意味着 class_num / 输出列布局不匹配
  std::map<std::string, int> hist;

  int print_every = opt.print_every;
  if (print_every <= 0) print_every = use_camera ? 30 : 1;

  auto process = [&](const cv::Mat & img, int index, const fs::path & save_path) {
    if (img.empty()) return false;

    const auto t0 = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, index);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    frames++;
    total_ms += ms;
    total_det += static_cast<int>(armors.size());
    if (!armors.empty()) frames_with_det++;
    for (const auto & a : armors) {
      hist[armor_name(a)]++;
      if (a.confidence > 1.5) bogus_conf++;
    }

    if (!quiet && index % print_every == 0) {
      fmt::print("[{:04d}] {:6.2f} ms  {} 个", index, ms, armors.size());
      for (const auto & a : armors) {
        fmt::print(" | {}", armor_name(a));
        fmt::print(" c={:.2f} box=({},{},{},{})", a.confidence, a.box.x, a.box.y, a.box.width,
                   a.box.height);
        if (a.points.size() == 4)
          fmt::print(" kpt=({:.0f},{:.0f})", a.points[0].x, a.points[0].y);
      }
      fmt::print("\n");
      std::fflush(stdout);
    }

    nlohmann::json frame;
    frame["index"] = index;
    frame["ms"] = ms;
    frame["detections"] = to_json(armors);
    report["frames"].push_back(std::move(frame));

    if (save_raw && index < 5) {
      fs::create_directories(output_dir);
      const auto raw_path = fs::path(output_dir) / fmt::format("raw_{:04d}.png", index);
      cv::imwrite(raw_path.string(), img);
      if (!quiet) fmt::print("  raw -> {}\n", raw_path.string());
    }

    cv::Mat annotated = img.clone();
    draw(annotated, armors);

    if (save) {
      if (stream_mode) {
        if (!writer.isOpened()) {
          const double fps =
            (is_video && video.get(cv::CAP_PROP_FPS) > 1) ? video.get(cv::CAP_PROP_FPS) : 30;
          writer.open(
            (fs::path(output_dir) / "annotated.avi").string(),
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, annotated.size());
        }
        writer.write(annotated);
      } else {
        cv::imwrite((fs::path(output_dir) / save_path.filename()).string(), annotated);
      }
    }

    if (show) {
      cv::imshow("detector_verify", annotated);
      const int key = cv::waitKey(1);
      if (key == 'q' || key == 27) return false;
    }
    return !(max_frames > 0 && frames >= max_frames);
  };

  int index = 0;
  if (use_camera) {
    // HikRobot::read() 会阻塞在内部队列上：用看门狗线程保证 5 秒内给出诊断并退出
    std::atomic<bool> got_first{false};
    std::thread watchdog([&got_first] {
      for (int i = 0; i < 50; ++i) {
        if (got_first) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      fmt::print(
        "\n相机 5 秒内没有输出帧。常见原因：\n"
        "  1) 当前用户无 USB 写权限（/dev/bus/usb/... 属 root:root）\n"
        "     echo 'SUBSYSTEM==\"usb\", ATTR{{idVendor}}==\"2bdf\", MODE=\"0666\", "
        "GROUP=\"plugdev\"' | sudo tee /etc/udev/rules.d/99-hikrobot.rules\n"
        "     sudo udevadm control --reload-rules && sudo udevadm trigger\n"
        "     然后重新插拔相机\n"
        "  2) 相机被其他进程占用（MVS 客户端 / 其他采集程序）\n"
        "  3) config 中 camera_name / vid_pid / exposure_ms 与实际不一致\n");
      std::fflush(stdout);
      std::_Exit(1);
    });
    watchdog.detach();

    cv::Mat first_img;
    std::chrono::steady_clock::time_point first_ts;
    camera->read(first_img, first_ts);
    got_first = true;

    bool keep_running = process(first_img, index, {});
    index++;

    tools::Exiter exiter;
    while (keep_running && !exiter.exit()) {
      cv::Mat img;
      std::chrono::steady_clock::time_point ts;
      camera->read(img, ts);
      if (img.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      keep_running = process(img, index, {});
      index++;
    }
  } else if (is_video) {
    cv::Mat img;
    while (video.read(img)) {
      if (!process(img, index, {})) break;
      index++;
    }
  } else {
    for (const auto & p : image_paths) {
      cv::Mat img = cv::imread(p.string());
      if (img.empty()) {
        fmt::print("读取失败: {}\n", p.string());
        continue;
      }
      if (!process(img, index, p)) break;
      index++;
    }
  }

  if (writer.isOpened()) writer.release();

  const double avg_ms = frames > 0 ? total_ms / frames : 0;
  report["summary"] = {
    {"frames", frames},
    {"frames_with_detection", frames_with_det},
    {"detections", total_det},
    {"avg_ms", avg_ms},
    {"fps", avg_ms > 0 ? 1000.0 / avg_ms : 0.0},
    {"class_histogram", hist}};

  fmt::print("\n===== 汇总 =====\n");
  fmt::print("帧数        : {}\n", frames);
  fmt::print("有检出帧数  : {} ({:.1f}%)\n", frames_with_det,
             frames > 0 ? 100.0 * frames_with_det / frames : 0.0);
  fmt::print("检测目标数  : {}\n", total_det);
  fmt::print("平均耗时    : {:.2f} ms ({:.1f} fps)\n", avg_ms, avg_ms > 0 ? 1000.0 / avg_ms : 0.0);
  for (const auto & [name, count] : hist) fmt::print("  {:<10} {}\n", name, count);
  if (save) fmt::print("结果目录    : {}\n", output_dir);

  if (bogus_conf > 0) {
    fmt::print(
      "\n⚠ 检测到 {} 个置信度 > 1.5 的目标：模型输出列布局与适配器不匹配。\n"
      "  适配器假设 [4 框 + class_num 类 + 4 关键点]，请核对：\n"
      "  - class_num 是否等于模型叠加通道数（去掉 4 框和 8 关键点）\n"
      "  - 模型是否真的是 4 个关键点（best2-sim 是 1 类 + 5 关键点，不兼容）\n",
      bogus_conf);
  }

  if (!json_path.empty()) {
    std::ofstream ofs(json_path);
    ofs << report.dump(1);
    fmt::print("JSON        : {}\n", json_path);
  }

  if (total_det == 0) {
    fmt::print(
      "\n提示：没有任何检出。请检查\n"
      "  1) config 中 yolo11_model_path 是否指向转换后的 .onnx\n"
      "  2) class_num 是否与模型输出通道一致（yolo11.xml -> 38）\n"
      "  3) 输入图像颜色/曝光是否正常（模型输入为 BGR -> RGB 且 /255）\n");
  }

  return 0;
}
