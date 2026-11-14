#include <Eigen/Geometry>
#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>
#include <vector>

#include "io/reconnecting_serial.hpp"
#include "io/vision_protocol.hpp"
#include "tools/exiter.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明 }"
  "{@config-path   | configs/sentry.yaml | yaml配置文件路径 }"
  "{port           |                     | 覆盖 com_port，例如 /dev/ttyACM0 }"
  "{no-display     | false               | 无窗口运行，保留终端和 UDP 曲线 }"
  "{frames         | 0                   | 收到指定数量的 CRC 正确帧后退出，0 为不限 }"
  "{plot-port      | 9870                | 本机 PlotJuggler UDP JSON 端口 }";

namespace
{
constexpr double degrees = 180.0 / CV_PI;

std::vector<io::GimbalToVision> decode_frames(std::vector<uint8_t> & bytes, uint64_t & crc_errors)
{
  std::vector<io::GimbalToVision> frames;
  while (bytes.size() >= 2) {
    const auto head = std::search(
      bytes.begin(), bytes.end(), std::begin(io::gimbal_struct_header),
      std::end(io::gimbal_struct_header));
    if (head == bytes.end()) {
      const bool keep_a = bytes.back() == io::gimbal_struct_header[0];
      bytes.clear();
      if (keep_a) bytes.push_back(io::gimbal_struct_header[0]);
      break;
    }
    bytes.erase(bytes.begin(), head);
    if (bytes.size() < sizeof(io::GimbalToVision)) break;
    if (!tools::check_crc16(bytes.data(), sizeof(io::GimbalToVision))) {
      ++crc_errors;
      bytes.erase(bytes.begin());
      continue;
    }
    io::GimbalToVision frame{};
    std::memcpy(&frame, bytes.data(), sizeof(frame));
    frames.push_back(frame);
    bytes.erase(bytes.begin(), bytes.begin() + sizeof(frame));
  }
  return frames;
}

struct Feedback
{
  io::GimbalToVision packet;
  Clock::time_point received = Clock::now();
  Eigen::Quaterniond orientation;
  double norm;
  std::optional<Eigen::Vector3d> ypr;

  explicit Feedback(const io::GimbalToVision & frame)
  : packet(frame), orientation(frame.q[0], frame.q[1], frame.q[2], frame.q[3]),
    norm(orientation.norm())
  {
    if (!orientation.coeffs().allFinite() || norm < 1e-6) return;
    orientation.normalize();
    const Eigen::Matrix3d rotation = orientation.toRotationMatrix();
    // The firmware supplies w,x,y,z for a body-to-world rotation.
    ypr = Eigen::Vector3d{
      std::atan2(rotation(1, 0), rotation(0, 0)),
      std::asin(std::clamp(-rotation(2, 0), -1.0, 1.0)),
      std::atan2(rotation(2, 1), rotation(2, 2))};
  }

  bool unit() const { return ypr.has_value() && std::abs(norm - 1.0) < 0.01; }
  double yaw_error() const { return std::remainder((*ypr)[0] - packet.yaw, 2.0 * CV_PI); }
  double pitch_error() const { return std::remainder((*ypr)[1] - packet.pitch, 2.0 * CV_PI); }
};

void draw_orientation(cv::Mat & image, const Eigen::Quaterniond & orientation)
{
  const cv::Point origin(835, 370);
  const auto project = [&](const Eigen::Vector3d & point) {
    return origin + cv::Point(
      cvRound(115.0 * (point.x() - point.y()) / std::sqrt(2.0)),
      cvRound(115.0 * ((point.x() + point.y()) / std::sqrt(8.0) - 0.8660254 * point.z())));
  };
  const auto rotation = orientation.toRotationMatrix();
  std::array<cv::Point, 8> box;
  for (int i = 0; i < 8; ++i) {
    const Eigen::Vector3d corner(
      i & 1 ? 0.6 : -0.6, i & 2 ? 0.6 : -0.6, i & 4 ? 0.6 : -0.6);
    box[i] = project(rotation * corner);
  }
  for (int i = 0; i < 8; ++i)
    for (int axis = 0; axis < 3; ++axis) {
      const int j = i ^ (1 << axis);
      if (i < j) cv::line(image, box[i], box[j], {200, 200, 200}, 2, cv::LINE_AA);
    }

  const std::array<cv::Scalar, 3> colors = {
    cv::Scalar(70, 70, 255), cv::Scalar(70, 230, 70), cv::Scalar(255, 140, 70)};
  for (int axis = 0; axis < 3; ++axis) {
    const auto world_end = project(Eigen::Vector3d::Unit(axis) * 1.7);
    cv::arrowedLine(image, origin, world_end, {100, 100, 100}, 1, cv::LINE_AA);
    cv::putText(image, fmt::format("W{}", "XYZ"[axis]), world_end,
      cv::FONT_HERSHEY_SIMPLEX, 0.5, {150, 150, 150}, 1, cv::LINE_AA);
    const auto body_end = project(rotation.col(axis) * 1.35);
    cv::arrowedLine(image, origin, body_end, colors[axis], 3, cv::LINE_AA);
    cv::putText(image, fmt::format("B{}", "XYZ"[axis]), body_end,
      cv::FONT_HERSHEY_SIMPLEX, 0.6, colors[axis], 2, cv::LINE_AA);
  }
}

cv::Mat panel(
  const std::optional<Feedback> & latest, bool fresh, const std::string & port,
  uint64_t frames, uint64_t crc_errors, double hz)
{
  cv::Mat image(680, 1120, CV_8UC3, cv::Scalar(25, 25, 30));
  int line = 32;
  const auto text = [&](const std::string & value, cv::Scalar color = {220, 220, 220}) {
    cv::putText(image, value, {24, line}, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv::LINE_AA);
    line += 32;
  };
  text("Gimbal raw quaternion feedback | RX ONLY | q / Esc to exit");
  text("Port: " + port);
  text(fmt::format("CRC-valid frames: {}   CRC errors: {}   RX: {:.1f} Hz", frames, crc_errors, hz));
  if (!latest) {
    text("WAITING FOR AB FEEDBACK", {70, 180, 255});
    return image;
  }
  const auto & data = *latest;
  const double age = std::chrono::duration<double, std::milli>(Clock::now() - data.received).count();
  text(fmt::format("{}   age: {:.1f} ms   mode: {}", fresh ? "LIVE" : "STALE / DISCONNECTED",
    age, data.packet.mode), fresh ? cv::Scalar(80, 230, 80) : cv::Scalar(70, 70, 255));
  text(fmt::format("raw w: {:+.7f}   x: {:+.7f}", data.packet.q[0], data.packet.q[1]));
  text(fmt::format("raw y: {:+.7f}   z: {:+.7f}", data.packet.q[2], data.packet.q[3]));
  text(fmt::format("|q|: {:.8f}   unit: {}", data.norm, data.unit() ? "YES" : "NO"));
  if (data.ypr) {
    text(fmt::format("q yaw:   {:+9.3f} deg", (*data.ypr)[0] * degrees));
    text(fmt::format("q pitch: {:+9.3f} deg", (*data.ypr)[1] * degrees));
    text(fmt::format("q roll:  {:+9.3f} deg", (*data.ypr)[2] * degrees));
    text(fmt::format("packet yaw/pitch: {:+.3f} / {:+.3f} deg",
      data.packet.yaw * degrees, data.packet.pitch * degrees));
    text(fmt::format("q - packet: {:+.4f} / {:+.4f} deg",
      data.yaw_error() * degrees, data.pitch_error() * degrees));
    if (fresh) draw_orientation(image, data.orientation);
  } else {
    text("INVALID QUATERNION - no Euler angles / 3D pose", {70, 70, 255});
  }
  text(fmt::format("bullet speed: {:.2f} m/s   count: {}",
    data.packet.bullet_speed, data.packet.bullet_count));
  line = 600;
  text("ZYX Euler, body -> world. Red X / Green Y / Blue Z.");
  text("Raw q is normalized only for Euler/3D display.");
  return image;
}

nlohmann::json plot_data(
  const Feedback & data, double t, uint64_t frames, uint64_t crc_errors, double hz, uint64_t generation)
{
  nlohmann::json result = {
    {"t", t}, {"fresh", 1}, {"rx_frames", frames}, {"crc_errors", crc_errors}, {"rx_hz", hz},
    {"serial_generation", generation}, {"age_ms", 0}, {"mode", data.packet.mode},
    {"q_w", data.packet.q[0]}, {"q_x", data.packet.q[1]},
    {"q_y", data.packet.q[2]}, {"q_z", data.packet.q[3]},
    {"quaternion_norm", data.norm}, {"q_norm_error", std::abs(data.norm - 1.0)},
    {"q_valid", data.ypr ? 1 : 0}, {"q_unit", data.unit() ? 1 : 0},
    {"feedback_yaw_deg", data.packet.yaw * degrees},
    {"feedback_pitch_deg", data.packet.pitch * degrees},
    {"yaw_rate_dps", data.packet.yaw_vel * degrees},
    {"pitch_rate_dps", data.packet.pitch_vel * degrees},
    {"bullet_speed", data.packet.bullet_speed}, {"bullet_count", data.packet.bullet_count}};
  if (data.ypr) {
    result["q_yaw_deg"] = (*data.ypr)[0] * degrees;
    result["q_pitch_deg"] = (*data.ypr)[1] * degrees;
    result["q_roll_deg"] = (*data.ypr)[2] * degrees;
    result["yaw_error_deg"] = data.yaw_error() * degrees;
    result["pitch_error_deg"] = data.pitch_error() * degrees;
    result["near_euler_singularity"] = std::abs(std::cos((*data.ypr)[1])) < 0.01 ? 1 : 0;
  }
  return result;
}
}  // namespace

int main(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto port = cli.get<std::string>("port");
  const auto config_path = cli.get<std::string>(0);
  const auto no_display = cli.get<bool>("no-display");
  const auto frame_limit = cli.get<int>("frames");
  const auto plot_port = cli.get<int>("plot-port");
  if (!cli.check() || frame_limit < 0 || plot_port < 1 || plot_port > 65535) {
    cli.printErrors();
    fmt::print(stderr, "frames 必须 >= 0，plot-port 必须在 1..65535\n");
    return 1;
  }
  if (port.empty()) port = tools::read<std::string>(tools::load(config_path), "com_port");

  tools::Exiter exiter;
  tools::Plotter plotter("127.0.0.1", static_cast<uint16_t>(plot_port));
  io::ReconnectingSerial serial(port, 115200, 10, "GimbalFeedback");
  fmt::print("只接收 AB 43 字节反馈，不发送控制。四元数按 w,x,y,z / B->W / ZYX 解释。\n");
  fmt::print("串口: {}；UDP JSON: 127.0.0.1:{}\n", port, plot_port);
  if (!no_display) cv::namedWindow("Gimbal feedback", cv::WINDOW_NORMAL);

  uint64_t frames = 0, crc_errors = 0, generation = 0, rate_frames = 0;
  double hz = 0;
  const auto start = Clock::now();
  auto rate_start = start, next_display = start, next_status = start;
  std::optional<Feedback> latest;
  std::vector<uint8_t> pending;
  while (!exiter.exit()) {
    if (serial.ensure_open()) {
      if (generation != serial.generation()) {
        generation = serial.generation();
        pending.clear();
        latest.reset();
        rate_frames = 0;
        hz = 0;
        rate_start = Clock::now();
      }
      std::array<uint8_t, 256> buffer;
      const auto count = serial.read(buffer.data(), buffer.size());
      if (serial.generation() != generation || !serial.connected()) {
        pending.clear();
        latest.reset();
        continue;
      }
      pending.insert(pending.end(), buffer.begin(), buffer.begin() + count);
      for (const auto & packet : decode_frames(pending, crc_errors)) {
        latest.emplace(packet);
        ++frames;
        ++rate_frames;
        plotter.plot(plot_data(*latest, std::chrono::duration<double>(Clock::now() - start).count(),
          frames, crc_errors, hz, generation));
      }
    } else {
      pending.clear();
      std::this_thread::sleep_for(10ms);
    }
    const auto now = Clock::now();
    const double rate_dt = std::chrono::duration<double>(now - rate_start).count();
    if (rate_dt >= 1.0) {
      hz = rate_frames / rate_dt;
      rate_frames = 0;
      rate_start = now;
    }
    const bool fresh = latest && serial.connected() && now - latest->received <= 500ms;
    if (now >= next_status) {
      if (fresh && latest->ypr) {
        fmt::print("RX {:.1f}Hz #{} CRCerr={} |q|={:.6f} YPR=({:+.2f},{:+.2f},{:+.2f})deg diff=({:+.4f},{:+.4f})deg\n",
          hz, frames, crc_errors, latest->norm, (*latest->ypr)[0] * degrees,
          (*latest->ypr)[1] * degrees, (*latest->ypr)[2] * degrees,
          latest->yaw_error() * degrees, latest->pitch_error() * degrees);
      } else {
        fmt::print("{}: frames={} CRCerr={}\n",
          fresh ? "收到帧，但四元数无效" : "等待反馈 / 反馈已超时", frames, crc_errors);
      }
      if (!fresh)
        plotter.plot({{"t", std::chrono::duration<double>(now - start).count()}, {"fresh", 0},
          {"rx_hz", hz}, {"rx_frames", frames}, {"crc_errors", crc_errors}});
      next_status = now + 100ms;
    }
    if (!no_display) {
      if (now >= next_display) {
        cv::imshow("Gimbal feedback", panel(latest, fresh, port, frames, crc_errors, hz));
        next_display = now + 33ms;
      }
      const auto key = cv::waitKey(1);
      if (key == 'q' || key == 'Q' || key == 27) break;
    }
    if (frame_limit > 0 && frames >= static_cast<uint64_t>(frame_limit)) break;
  }
  return 0;
}
