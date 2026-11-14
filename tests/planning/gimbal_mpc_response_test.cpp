#include <Eigen/Geometry>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/tinympc/tiny_api.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

namespace
{
using Clock = std::chrono::steady_clock;
using Pair = std::array<double, 2>;
using namespace std::chrono_literals;
constexpr double rad = CV_PI / 180.0;
constexpr double deg = 180.0 / CV_PI;
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
constexpr int horizon = 100;
constexpr int command_index = horizon / 2;
const cv::Scalar ink(225, 228, 235), muted(145, 150, 165);
const cv::Scalar ref_color(20, 215, 250), cmd_color(225, 80, 225), rx_color(245, 210, 65);

struct Params
{
  double width = 6 * rad;
  double height = 3 * rad;
  double frequency = 1;
  double rounding = 0.4;
  double ramp = 1;
  double max_velocity = 180 * rad;
  bool reverse = false;
};

struct Motion
{
  Pair p{}, v{}, a{};
};

Motion rectangle(const Params & p, double t)
{
  Motion out;
  const double w = p.width, h = p.height;
  const double direction = p.reverse ? -1.0 : 1.0;
  if (w < 1e-9 || h < 1e-9) {
    const double omega = direction * 2 * CV_PI * p.frequency;
    for (int axis = 0; axis < 2; ++axis) {
      const double amplitude = (axis == 0 ? w : h) / 2;
      out.p[axis] = amplitude * std::sin(omega * t);
      out.v[axis] = amplitude * omega * std::cos(omega * t);
      out.a[axis] = -amplitude * omega * omega * std::sin(omega * t);
    }
    return out;
  }
  const double r = std::min(w, h) / 2 * std::clamp(p.rounding, 0.01, 1.0);
  const double horizontal = std::max(0.0, w - 2 * r);
  const double vertical = std::max(0.0, h - 2 * r);
  const double arc = CV_PI * r / 2;
  const std::array<double, 8> lengths = {horizontal, arc, vertical, arc, horizontal, arc, vertical, arc};
  const double perimeter = 2 * (horizontal + vertical) + 4 * arc;
  const double speed = perimeter * p.frequency;
  double distance = std::fmod(direction * speed * t, perimeter);
  if (distance < 0) distance += perimeter;
  int segment = 0;
  while (segment < 7 && distance >= lengths[segment]) distance -= lengths[segment++];
  if ((segment & 1) == 0) {
    const std::array<Pair, 4> starts = {
      Pair{-w / 2 + r, -h / 2}, Pair{w / 2, -h / 2 + r},
      Pair{w / 2 - r, h / 2}, Pair{-w / 2, h / 2 - r}};
    const std::array<Pair, 4> directions = {Pair{1, 0}, Pair{0, 1}, Pair{-1, 0}, Pair{0, -1}};
    for (int axis = 0; axis < 2; ++axis) {
      out.p[axis] = starts[segment / 2][axis] + distance * directions[segment / 2][axis];
      out.v[axis] = direction * speed * directions[segment / 2][axis];
    }
  } else {
    const int corner = segment / 2;
    const std::array<Pair, 4> centers = {
      Pair{w / 2 - r, -h / 2 + r}, Pair{w / 2 - r, h / 2 - r},
      Pair{-w / 2 + r, h / 2 - r}, Pair{-w / 2 + r, -h / 2 + r}};
    const double angle = -CV_PI / 2 + corner * CV_PI / 2 + distance / r;
    const Pair normal{std::cos(angle), std::sin(angle)};
    out.p = {centers[corner][0] + r * normal[0], centers[corner][1] + r * normal[1]};
    out.v = {-direction * speed * normal[1], direction * speed * normal[0]};
    out.a = {-speed * speed / r * normal[0], -speed * speed / r * normal[1]};
  }
  return out;
}

Motion reference(const Params & p, double t, const Pair & start)
{
  if (t <= 0) return {start, {0, 0}, {0, 0}};
  Motion out = rectangle(p, t);
  if (t >= p.ramp) return out;
  const double u = t / p.ramp;
  const double gain = u * u * u * (10 + u * (-15 + 6 * u));
  const double velocity = 30 * u * u * (1 - u) * (1 - u) / p.ramp;
  const double acceleration = (60 * u - 180 * u * u + 120 * u * u * u) / (p.ramp * p.ramp);
  for (int axis = 0; axis < 2; ++axis) {
    const double delta = out.p[axis] - start[axis];
    out.a[axis] = gain * out.a[axis] + 2 * velocity * out.v[axis] + acceleration * delta;
    out.v[axis] = gain * out.v[axis] + velocity * delta;
    out.p[axis] = start[axis] + gain * delta;
  }
  return out;
}

struct SolverDeleter
{
  void operator()(TinySolver * p) const
  {
    if (!p) return;
    delete p->solution;
    delete p->settings;
    delete p->cache;
    delete p->work;
    delete p;
  }
};

struct AxisResult
{
  double p = missing, v = missing, a = missing;
  int iterations = 0;
  bool converged = false, limited = false;
  double primal = missing, dual = missing;
};

// Same reference-initialized, middle-of-horizon planning convention as auto_aim::Planner.
// The MCU closes the physical tracking loop; its measurements are never replaced by a plant model.
class AxisMpc
{
public:
  AxisMpc(const YAML::Node & yaml, const std::string & axis, double dt, int iterations)
  : dt_(dt), max_acc_(yaml["max_" + axis + "_acc"].as<double>())
  {
    const auto q = yaml["Q_" + axis].as<std::vector<double>>();
    const auto r = yaml["R_" + axis].as<std::vector<double>>();
    if (q.size() != 2 || r.size() != 1 || !std::isfinite(max_acc_) || max_acc_ <= 0 ||
        !std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(r[0]) ||
        q[0] <= 0 || q[1] < 0 || r[0] <= 0)
      throw std::runtime_error("Invalid MPC weights / acceleration bounds: " + axis);
    Eigen::Matrix2d A;
    A << 1, dt, 0, 1;
    Eigen::Vector2d B(0, dt), f(0, 0);
    Eigen::Matrix2d Q = Eigen::Vector2d(q[0], q[1]).asDiagonal();
    Eigen::Matrix<double, 1, 1> R;
    R << r[0];
    TinySolver * raw = nullptr;
    const int result = tiny_setup(&raw, A, B, f, Q, R, 1.0, 2, 1, horizon, 0);
    solver_.reset(raw);
    if (result != 0) throw std::runtime_error("TinyMPC setup failed: " + axis);
    solver_->settings->max_iter = iterations;
    solver_->settings->adaptive_rho = 0;
  }

  double max_acceleration() const { return max_acc_; }

  void reset()
  {
    auto & w = *solver_->work;
    w.x.setZero(); w.u.setZero(); w.q.setZero(); w.r.setZero();
    w.v.setZero(); w.vnew.setZero(); w.z.setZero(); w.znew.setZero();
    w.g.setZero(); w.y.setZero();
  }

  AxisResult solve(const Params & p, double phase, const Pair & start, int axis)
  {
    auto & w = *solver_->work;
    for (int i = 0; i < horizon; ++i) {
      const auto ref = reference(p, phase + (i - command_index) * dt_, start);
      w.Xref.col(i) << ref.p[axis], ref.v[axis];
    }
    const double half_span = (axis == 0 ? p.width : p.height) / 2;
    const double position_bound = std::max(half_span, std::abs(start[axis])) + 1 * rad;
    Eigen::MatrixXd lo(2, horizon), hi(2, horizon);
    lo.row(0).setConstant(-position_bound); hi.row(0).setConstant(position_bound);
    lo.row(1).setConstant(-p.max_velocity); hi.row(1).setConstant(p.max_velocity);
    tiny_set_bound_constraints(solver_.get(), lo, hi,
      Eigen::MatrixXd::Constant(1, horizon - 1, -max_acc_),
      Eigen::MatrixXd::Constant(1, horizon - 1, max_acc_));
    tiny_set_x0(solver_.get(), w.Xref.col(0));
    tiny_solve(solver_.get());
    AxisResult out;
    out.p = w.x(0, command_index);
    out.v = w.x(1, command_index);
    out.a = w.u(0, command_index);
    out.iterations = solver_->solution->iter;
    out.converged = solver_->solution->solved != 0;
    out.primal = std::max(w.primal_residual_state, w.primal_residual_input);
    out.dual = std::max(w.dual_residual_state, w.dual_residual_input);
    out.limited = std::abs(out.a) >= max_acc_ * 0.99 ||
      std::abs(out.v) >= p.max_velocity * 0.99 || std::abs(out.p) >= position_bound * 0.99;
    // Bound the actual transmitted fields even when the iteration budget is exhausted.
    out.p = std::clamp(out.p, -position_bound, position_bound);
    out.v = std::clamp(out.v, -p.max_velocity, p.max_velocity);
    out.a = std::clamp(out.a, -max_acc_, max_acc_);
    return out;
  }

private:
  double dt_, max_acc_;
  std::unique_ptr<TinySolver, SolverDeleter> solver_;
};

enum class State { waiting, preview, follow, paused, idle, stale_gui, invalid_mpc, pitch_limit, fault };

const char * state_name(State state)
{
  switch (state) {
    case State::waiting: return "WAIT: no fresh real feedback";
    case State::preview: return "RX ONLY: reference / MPC preview";
    case State::follow: return "TX FOLLOW: FIRE ALWAYS OFF";
    case State::paused: return "PAUSED: no control";
    case State::idle: return "TX STOP: board is not AUTO_AIM";
    case State::stale_gui: return "TX STOP: GUI heartbeat stale";
    case State::invalid_mpc: return "TX STOP: invalid MPC result";
    case State::pitch_limit: return "TX STOP: pitch outside Euler range";
    default: return "FAULT: see terminal log";
  }
}

struct Sample
{
  double t = 0, phase = 0;
  Motion ref, cmd{{missing, missing}, {missing, missing}, {missing, missing}};
  Motion rx{{missing, missing}, {missing, missing}, {missing, missing}};
  std::array<double, 4> raw_q{missing, missing, missing, missing};
  std::array<AxisResult, 2> solver;
  double solve_ms = 0, loop_ms = 0, jitter_ms = 0, age_ms = missing;
  double loop_hz = 0, rx_hz = 0, tx_hz = 0;
  uint64_t generation = 0, overruns = 0;
  int mode = -1;
  bool feedback_valid = false, acceleration_valid = false, plan_valid = false, control = false;
  State state = State::waiting;
};

struct UiInput
{
  Params params;
  bool paused = false;
  uint64_t revision = 0, recenter = 0, clear = 0;
  Clock::time_point heartbeat = Clock::now();
};

struct Snapshot
{
  Sample latest;
  std::vector<Sample> history;
  Pair center{};
  Pair max_acc{};
  Params params;
  bool centered = false;
};

class ResponseWorker
{
public:
  ResponseWorker(const std::string & config_path, const YAML::Node & yaml, Params params,
                 double rate, int iterations, bool enable_control)
  : gimbal_(config_path), yaw_(yaml, "yaw", 1 / rate, iterations),
    pitch_(yaml, "pitch", 1 / rate, iterations), dt_(1 / rate), enable_control_(enable_control)
  {
    input_.params = params;
    published_.params = params;
    published_.max_acc = {yaw_.max_acceleration(), pitch_.max_acceleration()};
    thread_ = std::thread([this] { run(); });
  }

  ~ResponseWorker() { stop(); }

  void stop()
  {
    quit_ = true;
    if (thread_.joinable()) thread_.join();
  }

  void update(const UiInput & input)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    input_ = input;
  }

  Snapshot snapshot()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot out = published_;
    out.history.assign(history_.begin(), history_.end());
    return out;
  }

private:
  void run()
  {
    try { loop(); }
    catch (const std::exception & e) {
      tools::logger()->error("[MPC response] 控制循环退出：{}", e.what());
      std::lock_guard<std::mutex> lock(mutex_);
      published_.latest.state = State::fault;
      published_.latest.control = false;
    }
    if (enable_control_) gimbal_.send(false, false, 0, 0, 0, 0, 0, 0);
  }

  void loop()
  {
    const auto origin = Clock::now();
    auto next = origin, previous_tick = origin, phase_origin = origin, rate_origin = origin;
    std::optional<io::GimbalFeedback> previous_rx;
    Pair filtered_acc{missing, missing}, center{}, start{};
    uint64_t generation = 0, revision = 0, recenter = 0, clear = 0, overruns = 0;
    uint64_t rx_count = 0, tx_count = 0, loop_count = 0;
    bool centered = false, was_fresh = false, was_paused = false, was_auto = false;
    double paused_phase = 0, rx_hz = 0, tx_hz = 0, loop_hz = 0;
    State previous_state = State::fault;
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(dt_));

    while (!quit_) {
      const auto tick = Clock::now();
      UiInput input;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        input = input_;
      }
      const auto feedback = gimbal_.feedback();
      const bool fresh = feedback && tick - feedback->received <= 100ms &&
        std::isfinite(feedback->state.yaw) && std::isfinite(feedback->state.pitch) &&
        std::isfinite(feedback->state.yaw_vel) && std::isfinite(feedback->state.pitch_vel);
      const bool auto_mode = fresh && feedback->mode == io::GimbalMode::AUTO_AIM;
      const bool generation_changed = fresh && generation != feedback->generation;
      const bool center_changed = fresh && (!centered || generation_changed || !was_fresh ||
        recenter != input.recenter || (enable_control_ && auto_mode && !was_auto));
      const bool params_changed = revision != input.revision;
      const bool restart = fresh && (center_changed || params_changed || (was_paused && !input.paused));
      if (center_changed) {
        center = {feedback->state.yaw, feedback->state.pitch};
        centered = true;
        filtered_acc = {missing, missing};
        previous_rx.reset();
      }
      if (restart) {
        start = {std::remainder(feedback->state.yaw - center[0], 2 * CV_PI),
          feedback->state.pitch - center[1]};
        phase_origin = tick;
        paused_phase = 0;
        yaw_.reset();
        pitch_.reset();
      }
      if (input.paused && !was_paused)
        paused_phase = std::chrono::duration<double>(tick - phase_origin).count();
      if (center_changed || params_changed || clear != input.clear) {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.clear();
      }
      revision = input.revision;
      recenter = input.recenter;
      clear = input.clear;
      if (fresh) generation = feedback->generation;

      Sample sample;
      sample.t = std::chrono::duration<double>(tick - origin).count();
      sample.phase = input.paused ? paused_phase : std::chrono::duration<double>(tick - phase_origin).count();
      sample.loop_ms = std::chrono::duration<double, std::milli>(tick - previous_tick).count();
      sample.jitter_ms = sample.loop_ms - dt_ * 1000;
      previous_tick = tick;
      sample.feedback_valid = fresh;
      if (feedback) {
        sample.mode = static_cast<int>(feedback->mode);
        sample.age_ms = std::chrono::duration<double, std::milli>(tick - feedback->received).count();
        sample.generation = feedback->generation;
        sample.raw_q = {feedback->q.w(), feedback->q.x(), feedback->q.y(), feedback->q.z()};
      }
      if (fresh) {
        sample.rx.p = {std::remainder(feedback->state.yaw - center[0], 2 * CV_PI),
          feedback->state.pitch - center[1]};
        sample.rx.v = {feedback->state.yaw_vel, feedback->state.pitch_vel};
        if (!previous_rx || previous_rx->received != feedback->received) {
          ++rx_count;
          if (previous_rx && previous_rx->generation == feedback->generation) {
            const double elapsed = std::chrono::duration<double>(feedback->received - previous_rx->received).count();
            if (elapsed > 0.001 && elapsed <= 0.1) {
              const Pair previous_velocity{previous_rx->state.yaw_vel, previous_rx->state.pitch_vel};
              const double alpha = elapsed / (0.03 + elapsed);
              for (int axis = 0; axis < 2; ++axis) {
                const double derivative = (sample.rx.v[axis] - previous_velocity[axis]) / elapsed;
                filtered_acc[axis] = std::isfinite(filtered_acc[axis])
                  ? filtered_acc[axis] + alpha * (derivative - filtered_acc[axis]) : derivative;
              }
            } else filtered_acc = {missing, missing};
          }
          previous_rx = feedback;
        }
        sample.rx.a = filtered_acc;
        sample.acceleration_valid = std::isfinite(filtered_acc[0]) && std::isfinite(filtered_acc[1]);
      }
      if (centered) {
        sample.ref = reference(input.params, sample.phase, start);
        if (input.paused) sample.ref.v = sample.ref.a = Pair{0, 0};
      }
      if (fresh && centered && !input.paused) {
        const auto solve_start = Clock::now();
        sample.solver[0] = yaw_.solve(input.params, sample.phase, start, 0);
        sample.solver[1] = pitch_.solve(input.params, sample.phase, start, 1);
        sample.solve_ms = std::chrono::duration<double, std::milli>(Clock::now() - solve_start).count();
        sample.plan_valid = true;
        for (int axis = 0; axis < 2; ++axis) {
          const auto & result = sample.solver[axis];
          sample.cmd.p[axis] = result.p;
          sample.cmd.v[axis] = result.v;
          sample.cmd.a[axis] = result.a;
          sample.plan_valid = sample.plan_valid && std::isfinite(result.p) &&
            std::isfinite(result.v) && std::isfinite(result.a);
        }
      }
      bool ui_current;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ui_current = !input_.paused && input_.revision == revision && input_.recenter == recenter &&
          Clock::now() - input_.heartbeat <= 1000ms;
      }
      const auto latest = gimbal_.feedback();
      const bool link_current = fresh && latest && latest->generation == generation &&
        Clock::now() - latest->received <= 100ms;
      if (!link_current) sample.state = State::waiting;
      else if (input.paused) sample.state = State::paused;
      else if (!enable_control_) sample.state = State::preview;
      else if (!ui_current) sample.state = State::stale_gui;
      else if (!auto_mode || latest->mode != io::GimbalMode::AUTO_AIM) sample.state = State::idle;
      else if (!sample.plan_valid) sample.state = State::invalid_mpc;
      else if (std::abs(center[1] + sample.cmd.p[1]) >= 89 * rad) sample.state = State::pitch_limit;
      else sample.state = State::follow;
      sample.control = sample.state == State::follow && !quit_;
      if (enable_control_) {
        if (sample.control)
          gimbal_.send(true, false,
            std::remainder(center[0] + sample.cmd.p[0], 2 * CV_PI), sample.cmd.v[0], sample.cmd.a[0],
            center[1] + sample.cmd.p[1], sample.cmd.v[1], sample.cmd.a[1]);
        else gimbal_.send(false, false, 0, 0, 0, 0, 0, 0);
        ++tx_count;
      }
      ++loop_count;
      const double rate_elapsed = std::chrono::duration<double>(tick - rate_origin).count();
      if (rate_elapsed >= 0.5) {
        rx_hz = rx_count / rate_elapsed;
        tx_hz = tx_count / rate_elapsed;
        loop_hz = loop_count / rate_elapsed;
        rx_count = tx_count = loop_count = 0;
        rate_origin = tick;
      }
      sample.rx_hz = rx_hz; sample.tx_hz = tx_hz; sample.loop_hz = loop_hz;
      next += period;
      if (Clock::now() > next) {
        ++overruns;
        next = Clock::now() + period;
      }
      sample.overruns = overruns;
      if (sample.state != previous_state) {
        tools::logger()->info("[MPC response] {}", state_name(sample.state));
        previous_state = sample.state;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(sample);
        while (!history_.empty() && sample.t - history_.front().t > 30) history_.pop_front();
        published_.latest = sample;
        published_.params = input.params;
        published_.center = center;
        published_.centered = centered;
      }
      was_fresh = fresh;
      was_auto = auto_mode;
      was_paused = input.paused;
      std::this_thread::sleep_until(next);
    }
  }

  io::Gimbal gimbal_;
  AxisMpc yaw_, pitch_;
  double dt_;
  bool enable_control_;
  std::mutex mutex_;
  UiInput input_;
  Snapshot published_;
  std::deque<Sample> history_;
  std::atomic<bool> quit_ = false;
  std::thread thread_;
};

void label(cv::Mat & image, const std::string & value, int x, int y,
           const cv::Scalar & color = ink, double scale = 0.48)
{
  cv::putText(image, value, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

struct Slider
{
  std::string title;
  cv::Rect bar;
  int value, low, high;
  double divisor;
};

enum Action { none, apply, pause, recenter, clear_stats, save_csv, save_png, reverse_path, quit };

struct Button
{
  cv::Rect bounds;
  std::string title;
  Action action;
};

struct Controls
{
  std::vector<Slider> sliders;
  std::vector<Button> buttons;
  cv::Point mouse{-1, -1};
  int dragging = -1;
  Action pending = none;
  bool reverse = false;

  explicit Controls(const Params & p, double history)
  {
    sliders = {
      {"Yaw width (deg)", {1360, 110, 440, 10}, cvRound(p.width * deg * 10), 0, 300, 10},
      {"Pitch height (deg)", {1360, 148, 440, 10}, cvRound(p.height * deg * 10), 0, 200, 10},
      {"Cycles / second (Hz)", {1360, 186, 440, 10}, cvRound(p.frequency * 10), 1, 100, 10},
      {"Corner radius (%)", {1360, 224, 440, 10}, cvRound(p.rounding * 100), 1, 100, 1},
      {"Transition ramp (s)", {1360, 262, 440, 10}, cvRound(p.ramp * 10), 1, 50, 10},
      {"MPC max velocity (deg/s)", {1360, 300, 440, 10}, cvRound(p.max_velocity * deg), 1, 720, 1},
      {"Visible history (s)", {1360, 338, 440, 10}, cvRound(history * 10), 10, 300, 10}};
    buttons = {
      {{1350, 365, 150, 30}, "Apply [Enter]", apply},
      {{1510, 365, 150, 30}, "Run/Pause [Space]", pause},
      {{1670, 365, 150, 30}, "Recenter [C]", recenter},
      {{1350, 405, 150, 30}, "Clear [R]", clear_stats},
      {{1510, 405, 150, 30}, "CSV [S] / PNG [P]", save_csv},
      {{1670, 405, 150, 30}, "Reverse [D]", reverse_path}};
    reverse = p.reverse;
  }

  Params params() const
  {
    Params p;
    p.width = sliders[0].value / 10.0 * rad;
    p.height = sliders[1].value / 10.0 * rad;
    p.frequency = sliders[2].value / 10.0;
    p.rounding = sliders[3].value / 100.0;
    p.ramp = sliders[4].value / 10.0;
    p.max_velocity = sliders[5].value * rad;
    p.reverse = reverse;
    return p;
  }

  double history() const { return sliders[6].value / 10.0; }

  static void mouse_callback(int event, int x, int y, int, void * context)
  {
    auto & ui = *static_cast<Controls *>(context);
    ui.mouse = {x, y};
    if (event == cv::EVENT_LBUTTONDOWN) {
      for (std::size_t i = 0; i < ui.sliders.size(); ++i) {
        const auto & bar = ui.sliders[i].bar;
        if (cv::Rect(bar.x - 8, bar.y - 18, bar.width + 16, 35).contains(ui.mouse)) {
          ui.dragging = static_cast<int>(i);
          break;
        }
      }
      for (const auto & button : ui.buttons)
        if (button.bounds.contains(ui.mouse)) ui.pending = button.action;
    }
    if (ui.dragging >= 0 && (event == cv::EVENT_MOUSEMOVE || event == cv::EVENT_LBUTTONDOWN)) {
      auto & slider = ui.sliders[ui.dragging];
      const double fraction = std::clamp((x - slider.bar.x) / static_cast<double>(slider.bar.width), 0.0, 1.0);
      slider.value = slider.low + cvRound(fraction * (slider.high - slider.low));
    }
    if (event == cv::EVENT_LBUTTONUP) ui.dragging = -1;
  }

  void draw(cv::Mat & image) const
  {
    label(image, "REFERENCE SETTINGS - Enter applies changes", 1350, 65, ref_color, 0.51);
    for (const auto & slider : sliders) {
      label(image, slider.title, slider.bar.x, slider.bar.y - 8);
      label(image, fmt::format("{:.1f}", slider.value / slider.divisor), slider.bar.x + 355, slider.bar.y - 8, ref_color);
      cv::rectangle(image, slider.bar, cv::Scalar(60, 64, 75), cv::FILLED);
      const int x = slider.bar.x + cvRound((slider.value - slider.low) *
        slider.bar.width / static_cast<double>(slider.high - slider.low));
      cv::circle(image, {x, slider.bar.y + 5}, 7, ref_color, cv::FILLED, cv::LINE_AA);
    }
    for (const auto & button : buttons) {
      cv::rectangle(image, button.bounds, cv::Scalar(65, 65, 80), 1);
      label(image, button.title, button.bounds.x + 7, button.bounds.y + 20, ink, 0.4);
    }
  }
};

double series(const Sample & s, int axis, int metric, int curve)
{
  if (metric == 2) {
    if (curve == 0) return s.feedback_valid ? (s.ref.p[axis] - s.rx.p[axis]) * deg : missing;
    if (curve == 1) return s.feedback_valid && s.plan_valid ? (s.cmd.p[axis] - s.rx.p[axis]) * deg : missing;
    return s.plan_valid ? (s.ref.p[axis] - s.cmd.p[axis]) * deg : missing;
  }
  const Motion & m = curve == 0 ? s.ref : (curve == 1 ? s.cmd : s.rx);
  const bool valid = curve == 1 ? s.plan_valid : s.feedback_valid;
  if (!valid || (curve == 2 && metric == 3 && !s.acceleration_valid)) return missing;
  return (metric == 0 ? m.p[axis] : (metric == 1 ? m.v[axis] : m.a[axis])) * deg;
}

void chart(cv::Mat & image, const cv::Rect & area, const Snapshot & snapshot, double window,
           int axis, int metric, const cv::Point & mouse)
{
  cv::rectangle(image, area, cv::Scalar(60, 63, 75));
  const std::array<std::string, 4> names = {"angle relative to center (deg)", "velocity (deg/s)",
    "error (deg): REF-RX / CMD-RX / REF-CMD", "acceleration (deg/s2); RX = d(v)/dt + LPF"};
  label(image, (axis == 0 ? "YAW " : "PITCH ") + names[metric], area.x + 10, area.y + 20, ink, 0.46);
  const cv::Rect plot(area.x + 62, area.y + 36, area.width - 76, area.height - 69);
  const double end = snapshot.latest.t, begin = std::max(0.0, end - window);
  double low = missing, high = missing;
  for (const auto & s : snapshot.history) {
    if (s.t < begin) continue;
    for (int c = 0; c < 3; ++c) {
      const double value = series(s, axis, metric, c);
      if (!std::isfinite(value)) continue;
      if (!std::isfinite(low)) low = high = value;
      low = std::min(low, value); high = std::max(high, value);
    }
  }
  if (!std::isfinite(low)) {
    label(image, "Waiting for live data", area.x + 100, area.y + 115, muted);
    return;
  }
  const double pad = std::max((high - low) * 0.10, metric == 3 ? 20.0 : (metric == 1 ? 1.0 : 0.1));
  low -= pad; high += pad;
  const auto point = [&](double t, double value) {
    return cv::Point(plot.x + cvRound((t - begin) / std::max(window, end - begin) * plot.width),
      plot.y + cvRound((high - value) / (high - low) * plot.height));
  };
  for (int i = 0; i <= 4; ++i) {
    const double value = low + (high - low) * i / 4;
    const auto p = point(begin, value);
    cv::line(image, p, {plot.x + plot.width, p.y}, {48, 50, 62});
    label(image, fmt::format("{:.1f}", value), area.x + 3, p.y + 4, muted, 0.36);
    const double time = begin + window * i / 4;
    const int x = plot.x + i * plot.width / 4;
    cv::line(image, {x, plot.y}, {x, plot.y + plot.height}, {40, 43, 54});
    label(image, fmt::format("{:.1f}", time), x - 12, area.y + area.height - 8, muted, 0.38);
  }
  const std::array<cv::Scalar, 3> colors{ref_color, cmd_color, rx_color};
  for (int c = 0; c < 3; ++c) {
    std::optional<cv::Point> previous;
    for (const auto & s : snapshot.history) {
      if (s.t < begin) continue;
      const double value = series(s, axis, metric, c);
      if (!std::isfinite(value)) { previous.reset(); continue; }
      auto p = point(s.t, value);
      if (previous) cv::line(image, *previous, p, colors[c], c == 1 ? 2 : 1, cv::LINE_AA);
      previous = p;
    }
  }
  if (plot.contains(mouse)) {
    const double wanted = begin + (mouse.x - plot.x) * window / plot.width;
    const auto it = std::lower_bound(snapshot.history.begin(), snapshot.history.end(), wanted,
      [](const Sample & s, double t) { return s.t < t; });
    if (it != snapshot.history.end()) {
      cv::line(image, {mouse.x, plot.y}, {mouse.x, plot.y + plot.height}, muted);
      label(image, fmt::format("t {:.3f}: {:+.3f} / {:+.3f} / {:+.3f}", it->t,
        series(*it, axis, metric, 0), series(*it, axis, metric, 1), series(*it, axis, metric, 2)),
        area.x + 75, area.y + area.height - 8, ink, 0.4);
    }
  }
}

void xy_plot(cv::Mat & image, const Snapshot & snapshot, double window)
{
  const cv::Rect area(1350, 470, 470, 310);
  cv::rectangle(image, area, cv::Scalar(60, 63, 75));
  label(image, "Yaw / pitch path (deg, relative to center)", area.x + 10, area.y + 20);
  double extent_x = std::max(1.0, snapshot.params.width * deg / 2 + 0.5);
  double extent_y = std::max(1.0, snapshot.params.height * deg / 2 + 0.5);
  for (const auto & s : snapshot.history) {
    if (snapshot.latest.t - s.t > window) continue;
    if (s.feedback_valid) {
      extent_x = std::max(extent_x, std::abs(s.rx.p[0] * deg));
      extent_y = std::max(extent_y, std::abs(s.rx.p[1] * deg));
    }
  }
  const cv::Point origin(area.x + area.width / 2, area.y + 155);
  const double scale = std::min((area.width - 70) / (extent_x * 2), (area.height - 65) / (extent_y * 2));
  const auto point = [&](const Pair & p) {
    return origin + cv::Point(cvRound(p[0] * deg * scale), cvRound(p[1] * deg * scale));
  };
  cv::line(image, {area.x + 20, origin.y}, {area.x + area.width - 20, origin.y}, muted);
  cv::line(image, {origin.x, area.y + 35}, {origin.x, area.y + area.height - 25}, muted);
  std::optional<cv::Point> previous;
  for (int i = 0; i <= 200; ++i) {
    const auto p = point(rectangle(snapshot.params, i / (200.0 * snapshot.params.frequency)).p);
    if (previous) cv::line(image, *previous, p, ref_color, 1, cv::LINE_AA);
    previous = p;
  }
  for (int c = 0; c < 2; ++c) {
    previous.reset();
    for (const auto & s : snapshot.history) {
      if (snapshot.latest.t - s.t > window) continue;
      if (c == 0 ? !s.plan_valid : !s.feedback_valid) { previous.reset(); continue; }
      const auto p = point(c == 0 ? s.cmd.p : s.rx.p);
      if (previous) cv::line(image, *previous, p, c == 0 ? cmd_color : rx_color, 2, cv::LINE_AA);
      previous = p;
    }
    if (previous) cv::circle(image, *previous, 4, c == 0 ? cmd_color : rx_color, cv::FILLED);
  }
  label(image, "+yaw ->   pitch negative = up", area.x + 12, area.y + area.height - 7, muted, 0.4);
}

struct Statistics
{
  Pair rmse{}, peak{}, bias{}, servo{};
  std::size_t count = 0;
};

Statistics statistics(const Snapshot & view, double window)
{
  Statistics result;
  for (const auto & s : view.history) {
    if (!s.feedback_valid || !s.plan_valid || view.latest.t - s.t > window) continue;
    ++result.count;
    for (int axis = 0; axis < 2; ++axis) {
      const double error = (s.ref.p[axis] - s.rx.p[axis]) * deg;
      const double servo_error = (s.cmd.p[axis] - s.rx.p[axis]) * deg;
      result.rmse[axis] += error * error;
      result.servo[axis] += servo_error * servo_error;
      result.bias[axis] += error;
      result.peak[axis] = std::max(result.peak[axis], std::abs(error));
    }
  }
  for (int axis = 0; axis < 2; ++axis) {
    if (result.count == 0) {
      result.rmse[axis] = result.servo[axis] = result.bias[axis] = result.peak[axis] = missing;
    } else {
      result.rmse[axis] = std::sqrt(result.rmse[axis] / result.count);
      result.servo[axis] = std::sqrt(result.servo[axis] / result.count);
      result.bias[axis] /= result.count;
    }
  }
  return result;
}

cv::Mat render(const Snapshot & view, const Controls & controls, bool enable_control,
               double requested_rate, const std::string & notice)
{
  cv::Mat image(1120, 1840, CV_8UC3, cv::Scalar(23, 25, 32));
  const auto & s = view.latest;
  label(image, std::string("GIMBAL MPC RESPONSE | ") + state_name(s.state) +
    " | Space pause | q/Esc exit", 16, 28, s.control ? cmd_color : ref_color, 0.64);
  label(image, fmt::format("Loop {:.1f}/{:.0f} Hz | RX {:.1f} Hz | TX requests {:.1f} Hz | "
    "MPC {:.2f} ms | jitter {:+.2f} ms | overruns {}",
    s.loop_hz, requested_rate, s.rx_hz, s.tx_hz, s.solve_ms, s.jitter_ms, s.overruns),
    16, 52, ink, 0.47);
  label(image, "YELLOW reference | MAGENTA MPC | CYAN real feedback | Angle and XY axes are relative to the latched center",
    16, 76, muted, 0.44);
  for (int metric = 0; metric < 4; ++metric)
    for (int axis = 0; axis < 2; ++axis)
      chart(image, {16 + axis * 660, 94 + metric * 245, 644, 230}, view,
        controls.history(), axis, metric, controls.mouse);
  controls.draw(image);
  label(image, fmt::format("Active {:.1f} x {:.1f} deg, {:.1f} Hz | {}",
    view.params.width * deg, view.params.height * deg, view.params.frequency,
    view.params.reverse ? "REVERSE" : "FORWARD"), 1350, 457, ref_color, 0.45);
  xy_plot(image, view, controls.history());
  const auto stats = statistics(view, controls.history());
  int y = 802;
  const auto row = [&](const std::string & value, const cv::Scalar & color = ink) {
    label(image, value, 1350, y, color, 0.43);
    y += 21;
  };
  row("Absolute angles: REF / MPC / RX (deg)", muted);
  for (int axis = 0; axis < 2; ++axis) {
    const double center = view.centered ? view.center[axis] : missing;
    row(fmt::format("{} {:+.3f} / {:+.3f} / {:+.3f}", axis == 0 ? "Yaw  " : "Pitch",
      (center + s.ref.p[axis]) * deg, (center + s.cmd.p[axis]) * deg,
      (center + s.rx.p[axis]) * deg));
  }
  row(fmt::format("REF-RX RMSE Y/P {:.3f} / {:.3f} deg", stats.rmse[0], stats.rmse[1]), ref_color);
  row(fmt::format("Peak Y/P {:.3f} / {:.3f} | bias {:+.3f}/{:+.3f}",
    stats.peak[0], stats.peak[1], stats.bias[0], stats.bias[1]));
  row(fmt::format("CMD-RX RMSE Y/P {:.3f} / {:.3f} deg", stats.servo[0], stats.servo[1]), cmd_color);
  row(fmt::format("MPC iter {}/{} | converged {}/{} | limits {}/{}",
    s.solver[0].iterations, s.solver[1].iterations,
    s.solver[0].converged ? "Y" : "N", s.solver[1].converged ? "Y" : "N",
    s.solver[0].limited ? "Y" : "N", s.solver[1].limited ? "Y" : "N"));
  row(fmt::format("Residual primal {:.2e}/{:.2e} dual {:.2e}/{:.2e}",
    s.solver[0].primal, s.solver[1].primal, s.solver[0].dual, s.solver[1].dual), muted);
  row(fmt::format("Max accel Y/P {:.1f} / {:.1f} rad/s2", view.max_acc[0], view.max_acc[1]));
  row(fmt::format("RX age {:.1f} ms | mode {} | generation {}", s.age_ms, s.mode, s.generation));
  const Eigen::Quaterniond q(s.raw_q[0], s.raw_q[1], s.raw_q[2], s.raw_q[3]);
  if (s.feedback_valid && q.coeffs().allFinite() && q.norm() > 1e-6) {
    const Eigen::Matrix3d r = q.normalized().toRotationMatrix();
    row(fmt::format("Raw q YPR {:+.2f} / {:+.2f} / {:+.2f} deg",
      std::atan2(r(1, 0), r(0, 0)) * deg,
      std::asin(std::clamp(-r(2, 0), -1.0, 1.0)) * deg,
      std::atan2(r(2, 1), r(2, 2)) * deg), rx_color);
    row(fmt::format("q wxyz {:+.4f} {:+.4f} {:+.4f} {:+.4f}",
      q.w(), q.x(), q.y(), q.z()), muted);
  } else {
    row("Raw quaternion: N/A", muted);
    row("No fabricated feedback / plant model", muted);
  }
  row(fmt::format("{} | stats {} samples / {:.1f} s", enable_control ? "FOLLOW ENABLED" : "RX ONLY",
    stats.count, controls.history()), enable_control ? cmd_color : muted);
  row("FIRE = FALSE on every transmitted packet", cmd_color);
  label(image, notice.empty() ? "Drag sliders, Enter applies | C recenter | R clear | S CSV | P PNG | D reverse" : notice,
    16, 1105, ink, 0.45);
  return image;
}

std::filesystem::path output_path(const std::filesystem::path & folder, const std::string & extension)
{
  std::filesystem::create_directories(folder);
  const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  return folder / fmt::format("response_{}.{}", stamp, extension);
}

void save_history(const Snapshot & snapshot, const std::filesystem::path & path)
{
  std::ofstream file;
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file.open(path);
  file << std::setprecision(12);
  file << "t_s,phase_s,feedback_valid,plan_valid,control,mode,generation,state,center_yaw_deg,center_pitch_deg";
  for (const auto * axis : {"yaw", "pitch"})
    for (const auto * name : {"ref_deg", "mpc_deg", "rx_deg", "ref_vel_dps", "mpc_vel_dps", "rx_vel_dps",
                              "ref_acc_dps2", "mpc_acc_dps2", "rx_acc_lpf_dps2", "ref_rx_error_deg", "cmd_rx_error_deg"})
      file << ',' << axis << '_' << name;
  file << ",q_w,q_x,q_y,q_z,mpc_ms,loop_ms,jitter_ms,rx_age_ms,loop_hz,rx_hz,tx_request_hz,overruns"
          ",yaw_iter,pitch_iter,yaw_converged,pitch_converged,yaw_limited,pitch_limited"
          ",yaw_primal,pitch_primal,yaw_dual,pitch_dual,width_deg,height_deg,frequency_hz,corner_fraction\n";
  for (const auto & s : snapshot.history) {
    file << s.t << ',' << s.phase << ',' << s.feedback_valid << ',' << s.plan_valid << ',' << s.control
         << ',' << s.mode << ',' << s.generation << ',' << state_name(s.state)
         << ',' << snapshot.center[0] * deg << ',' << snapshot.center[1] * deg;
    for (int axis = 0; axis < 2; ++axis) {
      const double offset = snapshot.center[axis];
      file << ',' << (offset + s.ref.p[axis]) * deg << ',' << (offset + s.cmd.p[axis]) * deg
           << ',' << (offset + s.rx.p[axis]) * deg;
      for (int metric : {1, 3})
        for (int c = 0; c < 3; ++c) file << ',' << series(s, axis, metric, c);
      file << ',' << series(s, axis, 2, 0) << ',' << series(s, axis, 2, 1);
    }
    for (const auto value : s.raw_q) file << ',' << value;
    file << ',' << s.solve_ms << ',' << s.loop_ms << ',' << s.jitter_ms << ',' << s.age_ms
         << ',' << s.loop_hz << ',' << s.rx_hz << ',' << s.tx_hz << ',' << s.overruns;
    for (int field = 0; field < 5; ++field)
      for (int axis = 0; axis < 2; ++axis) {
        const auto & result = s.solver[axis];
        if (field == 0) file << ',' << result.iterations;
        if (field == 1) file << ',' << result.converged;
        if (field == 2) file << ',' << result.limited;
        if (field == 3) file << ',' << result.primal;
        if (field == 4) file << ',' << result.dual;
      }
    file << ',' << snapshot.params.width * deg << ',' << snapshot.params.height * deg << ','
         << snapshot.params.frequency << ',' << snapshot.params.rounding << '\n';
  }
}

Action key_action(int key)
{
  if (key == 'q' || key == 'Q' || key == 27) return quit;
  if (key == 10 || key == 13) return apply;
  if (key == ' ') return pause;
  if (key == 'c' || key == 'C') return recenter;
  if (key == 'r' || key == 'R') return clear_stats;
  if (key == 's' || key == 'S') return save_csv;
  if (key == 'p' || key == 'P') return save_png;
  if (key == 'd' || key == 'D') return reverse_path;
  return none;
}

const std::string keys =
  "{help h usage ? |                     | 输出参数说明 }"
  "{@config-path   | configs/sentry.yaml | Gimbal AB 串口及现有 MPC 参数 }"
  "{enable-control | true                | 开启真实云台跟随，开火始终关闭 }"
  "{width          | 6                   | 矩形完整 yaw 宽度，0..30 度 }"
  "{height         | 3                   | 矩形完整 pitch 高度，0..20 度 }"
  "{frequency      | 1                   | 完整一圈的频率，0.1..10 Hz }"
  "{rate           | 200                 | MPC/发送循环目标频率，50..500 Hz }"
  "{rounding       | 40                  | 圆角半径占短边一半的百分比，1..100 }"
  "{ramp           | 1                   | 轨迹展开/切换过渡时间，0.1..5 秒 }"
  "{max-velocity   | 180                 | 两轴规划速度上限，1..720 度/秒 }"
  "{iterations     | 10                  | TinyMPC 单轴最大迭代数，1..100 }"
  "{history        | 6                   | GUI 波形显示时长，1..30 秒 }"
  "{reverse        | false               | 反向绕行 }"
  "{output-folder  | out/gimbal_mpc_response | CSV 和波形截图保存目录 }";

int run(int argc, char ** argv)
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) { cli.printMessage(); return 0; }
  const auto config_path = cli.get<std::string>(0);
  const auto enable_control = cli.get<bool>("enable-control");
  const double width = cli.get<double>("width"), height = cli.get<double>("height");
  const double frequency = cli.get<double>("frequency"), rate = cli.get<double>("rate");
  const double rounding = cli.get<double>("rounding"), ramp = cli.get<double>("ramp");
  const double velocity = cli.get<double>("max-velocity"), history = cli.get<double>("history");
  const int iterations = cli.get<int>("iterations");
  const std::filesystem::path output_folder(cli.get<std::string>("output-folder"));
  if (!cli.check()) { cli.printErrors(); return 1; }
  const std::array<double, 8> numeric{width, height, frequency, rate, rounding, ramp, velocity, history};
  if (!std::all_of(numeric.begin(), numeric.end(), [](double x) { return std::isfinite(x); }) ||
      width < 0 || width > 30 || height < 0 || height > 20 || frequency < 0.1 || frequency > 10 ||
      rate < 50 || rate > 500 || rounding < 1 || rounding > 100 || ramp < 0.1 || ramp > 5 ||
      velocity < 1 || velocity > 720 || history < 1 || history > 30 || iterations < 1 || iterations > 100)
    throw std::runtime_error("参数超出范围，请用 --help 查看；角度单位为度，frequency 表示每秒完整圈数");
  Params params;
  params.width = width * rad; params.height = height * rad;
  params.frequency = frequency; params.rounding = rounding / 100;
  params.ramp = ramp; params.max_velocity = velocity * rad; params.reverse = cli.get<bool>("reverse");
  const auto yaml = YAML::LoadFile(config_path);
  tools::Exiter exiter;
  ResponseWorker worker(config_path, yaml, params, rate, iterations, enable_control);
  Controls controls(params, history);
  UiInput input;
  input.params = params;
  const std::string window = enable_control ? "Gimbal MPC response - FOLLOW / NO FIRE" : "Gimbal MPC response - RX ONLY";
  cv::namedWindow(window, cv::WINDOW_NORMAL);
  cv::resizeWindow(window, 1472, 896);
  cv::setMouseCallback(window, Controls::mouse_callback, &controls);
  tools::logger()->info("[MPC response] {}；构造矩形参考，MPC {} Hz，始终不开火",
    enable_control ? "已启用真实跟随" : "仅预览规划和真实反馈", rate);
  bool query_enabled = true, window_observed = false, window_closed = false;
  std::string notice;
  while (!exiter.exit()) {
    input.heartbeat = Clock::now();
    worker.update(input);
    const auto snapshot = worker.snapshot();
    auto image = render(snapshot, controls, enable_control, rate, notice);
    cv::imshow(window, image);
    const auto action_from_key = key_action(cv::waitKey(15));
    const auto action = action_from_key != none ? action_from_key : controls.pending;
    controls.pending = none;
    if (action == quit) break;
    if (action == apply) {
      input.params = controls.params();
      ++input.revision;
      notice = "Applied reference settings; transition restarts from the actual position";
    }
    if (action == pause) { input.paused = !input.paused; notice = input.paused ? "Paused; control released" : "Resuming with a smooth transition"; }
    if (action == recenter) { ++input.recenter; notice = "Recenter requested at the next fresh real feedback"; }
    if (action == clear_stats) { ++input.clear; notice = "Waveform history and window statistics cleared"; }
    if (action == reverse_path) { controls.reverse = !controls.reverse; notice = "Direction changed in pending settings; press Enter to apply"; }
    if (action == save_csv || action == save_png) {
      try {
        const auto path = output_path(output_folder, action == save_csv ? "csv" : "png");
        if (action == save_csv) save_history(snapshot, path);
        else if (!cv::imwrite(path.string(), image)) throw std::runtime_error("Failed to write PNG");
        notice = "Saved: " + path.string();
        tools::logger()->info("[MPC response] {}", notice);
      } catch (const std::exception & e) {
        notice = std::string("Save failed: ") + e.what();
        tools::logger()->error("[MPC response] {}", notice);
      }
    }
    if (query_enabled) {
      try {
        const double exists = cv::getWindowProperty(window, cv::WND_PROP_AUTOSIZE);
        if (std::isfinite(exists) && exists >= 0) window_observed = true;
        else if (window_observed) { window_closed = true; break; }
        else query_enabled = false;
      } catch (const cv::Exception &) {
        if (window_observed) { window_closed = true; break; }
        query_enabled = false;
      }
    }
  }
  worker.stop();
  if (!window_closed) cv::destroyWindow(window);
  tools::logger()->info("[MPC response] 已退出{}", enable_control ? "，已请求停控、不开火" : "（仅接收模式）");
  return 0;
}
}  // namespace

int main(int argc, char ** argv)
{
  try { return run(argc, argv); }
  catch (const std::exception & e) {
    tools::logger()->error("[MPC response] {}", e.what());
    return 1;
  }
}
