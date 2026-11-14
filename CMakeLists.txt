cmake_minimum_required(VERSION 3.16.3)
project(sp_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_LEGACY_CBOARD "Build deprecated C-board CAN/UART transport" OFF)
option(BUILD_TESTING "Build hardware-free contract tests with the default targets" OFF)
option(ENABLE_OPENVINO "Enable OpenVINO IR (.xml) inference backend" ON)
option(ENABLE_CUDA_DNN "Enable OpenCV DNN CUDA backend for ONNX models" ON)
include(CTest)
# Refresh CTest discovery even when BUILD_TESTING is switched off in an
# existing build directory, so old test registrations cannot linger.
enable_testing()

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

if(ENABLE_OPENVINO)
  if(NOT OpenVINO_DIR AND EXISTS "/opt/intel/openvino_2024.6.0/runtime/cmake")
    set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
  endif()
  find_package(OpenVINO QUIET COMPONENTS Runtime)
endif()

if(ENABLE_OPENVINO AND OpenVINO_FOUND)
  add_compile_definitions(SPVISION_WITH_OPENVINO)
  message(STATUS "Inference backend: OpenVINO ${OpenVINO_VERSION} (.xml) + OpenCV DNN (.onnx)")
else()
  message(STATUS "Inference backend: OpenCV DNN (.onnx) only")
endif()

if(ENABLE_CUDA_DNN)
  add_compile_definitions(SPVISION_WITH_CUDA_DNN)
  message(STATUS "OpenCV DNN CUDA enabled for device: GPU")
endif()

# All ROS consumers use the same dependency check, including geometry_msgs.
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(geometry_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)
set(SPVISION_WITH_ROS2 OFF)
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND geometry_msgs_FOUND
   AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
  set(SPVISION_WITH_ROS2 ON)
  message(STATUS "ROS2/sp_msgs found: sentry and manual ROS2 test targets enabled")
else()
  message(STATUS "Skipping sentry/ROS2 targets: source ROS2 and the built sp_msgs workspace to enable them")
endif()

include_directories(${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
if(SPVISION_WITH_ROS2)
  add_subdirectory(tasks/omniperception)
  add_subdirectory(tasks/sentry_decision)
endif()

# Only sentry and MPC entry points are part of the default application build.
add_subdirectory(src)

# Calibration and hardware/interactive tests are built by explicit target only,
# in both Debug and Release. Archived sources have no registered targets.
add_subdirectory(calibration EXCLUDE_FROM_ALL)
add_subdirectory(tests)

if(BUILD_TESTING)
  add_subdirectory(tests/contracts)
endif()
