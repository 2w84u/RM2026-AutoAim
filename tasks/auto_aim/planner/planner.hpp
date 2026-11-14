#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <array>
#include <list>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

struct PlannerTrace
{
  // Absolute yaw/pitch in rad, velocities in rad/s; HALF_HORIZON is the command time.
  Trajectory reference;
  Trajectory solution;
  std::array<Eigen::Vector4d, HORIZON> aim_xyza;
  double bullet_speed;
  double fly_time;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  Planner(const std::string & config_path, bool collect_trace = false);

  const std::optional<PlannerTrace> & trace() const { return trace_; }

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double default_bullet_speed_;
  bool collect_trace_;
  std::optional<PlannerTrace> trace_;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(
    Target & target, double yaw0, double bullet_speed,
    std::array<Eigen::Vector4d, HORIZON> * aim_points);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
