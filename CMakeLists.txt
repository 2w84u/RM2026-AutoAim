cmake_minimum_required(VERSION 3.16.3)
project(sp_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_LEGACY_CBOARD "Build deprecated C-board CAN/UART transport" OFF)
option(BUILD_TESTING "Build hardware-free contract tests with the default targets" OFF)
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

# Select an installed inference runtime automatically. The old cache switches
# are removed so reconfiguring an existing build uses the same discovery rules.
unset(ENABLE_OPENVINO CACHE)
unset(ENABLE_TENSORRT CACHE)

file(GLOB _spvision_openvino_hints LIST_DIRECTORIES true
  "/opt/intel/openvino*/runtime/cmake"
  "/opt/intel/openvino*/cmake"
  "/usr/local/lib/python*/dist-packages/openvino/cmake"
  "/usr/local/lib/python*/site-packages/openvino/cmake"
  "$ENV{HOME}/.local/lib/python*/site-packages/openvino/cmake")
if(DEFINED ENV{VIRTUAL_ENV})
  file(GLOB _spvision_venv_openvino_hints LIST_DIRECTORIES true
    "$ENV{VIRTUAL_ENV}/lib/python*/site-packages/openvino/cmake")
  list(APPEND _spvision_openvino_hints ${_spvision_venv_openvino_hints})
endif()
find_package(OpenVINO QUIET CONFIG COMPONENTS Runtime HINTS ${_spvision_openvino_hints})

find_package(CUDAToolkit QUIET)
if(TENSORRT_INCLUDE_DIR AND NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
  unset(TENSORRT_INCLUDE_DIR CACHE)
endif()
if(TENSORRT_LIBRARY AND NOT EXISTS "${TENSORRT_LIBRARY}")
  unset(TENSORRT_LIBRARY CACHE)
endif()
set(_spvision_tensorrt_hints /usr/local/TensorRT /usr/local/tensorrt /opt/tensorrt)
if(DEFINED ENV{TENSORRT_ROOT})
  list(PREPEND _spvision_tensorrt_hints "$ENV{TENSORRT_ROOT}")
endif()
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h
  HINTS "/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}" /usr/include ${_spvision_tensorrt_hints}
  PATH_SUFFIXES include "include/${CMAKE_LIBRARY_ARCHITECTURE}")
find_library(TENSORRT_LIBRARY NAMES nvinfer
  HINTS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" /usr/lib ${_spvision_tensorrt_hints}
  PATH_SUFFIXES lib lib64 "lib/${CMAKE_LIBRARY_ARCHITECTURE}")

set(_spvision_openvino_available OFF)
if(OpenVINO_FOUND AND TARGET openvino::runtime)
  set(_spvision_openvino_available ON)
endif()
set(_spvision_tensorrt_available OFF)
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND CUDAToolkit_FOUND AND TARGET CUDA::cudart)
  set(_spvision_tensorrt_available ON)
endif()

set(SPVISION_USE_OPENVINO OFF)
set(SPVISION_USE_TENSORRT OFF)
if(EXISTS "/etc/nv_tegra_release" AND _spvision_tensorrt_available)
  set(SPVISION_USE_TENSORRT ON)
elseif(_spvision_openvino_available)
  set(SPVISION_USE_OPENVINO ON)
elseif(_spvision_tensorrt_available)
  set(SPVISION_USE_TENSORRT ON)
else()
  message(FATAL_ERROR
    "No usable inference backend found. Install OpenVINO Runtime or TensorRT development "
    "files and the CUDA Toolkit, then configure again.\n"
    "OpenVINO Runtime found: ${_spvision_openvino_available}; "
    "NvInfer.h: '${TENSORRT_INCLUDE_DIR}'; libnvinfer: '${TENSORRT_LIBRARY}'; "
    "CUDA Toolkit found: ${CUDAToolkit_FOUND}.")
endif()

if(SPVISION_USE_TENSORRT)
  add_compile_definitions(SPVISION_WITH_TENSORRT)
  message(STATUS "Inference backend: TensorRT (${TENSORRT_LIBRARY}) + CUDA ${CUDAToolkit_VERSION}")
else()
  add_compile_definitions(SPVISION_WITH_OPENVINO)
  message(STATUS "Inference backend: OpenVINO ${OpenVINO_VERSION}")
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
