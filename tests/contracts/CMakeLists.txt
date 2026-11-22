cmake_minimum_required(VERSION 3.16)

# This directory can also be configured on its own, without camera SDKs,
# inference backends, Ceres, or ROS2. The same targets serve both build paths.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(vision_contract_tests LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  enable_testing()
  find_package(OpenCV REQUIRED COMPONENTS core)
  find_package(Eigen3 REQUIRED)
  find_package(fmt REQUIRED)
  find_package(spdlog REQUIRED)
  find_package(yaml-cpp REQUIRED)
  find_package(nlohmann_json REQUIRED)
endif()

get_filename_component(VISION_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(armor_geometry_contract_test armor_geometry_contract_test.cpp)
target_include_directories(armor_geometry_contract_test PRIVATE
  ${VISION_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(armor_geometry_contract_test PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
add_test(NAME armor_geometry_contract COMMAND armor_geometry_contract_test)

add_executable(nav_protocol_contract_test nav_protocol_contract_test.cpp ${VISION_SOURCE_DIR}/tools/crc.cpp)
target_include_directories(nav_protocol_contract_test PRIVATE ${VISION_SOURCE_DIR})
add_test(NAME nav_protocol_contract COMMAND nav_protocol_contract_test)

if(UNIX AND NOT APPLE)
  if(NOT TARGET serial)
    add_subdirectory(${VISION_SOURCE_DIR}/io/serial ${CMAKE_CURRENT_BINARY_DIR}/serial)
  endif()
  add_executable(serial_hotplug_contract_test serial_hotplug_contract_test.cpp)
  target_include_directories(serial_hotplug_contract_test PRIVATE ${VISION_SOURCE_DIR})
  target_link_libraries(serial_hotplug_contract_test PRIVATE serial util)
  add_test(NAME serial_hotplug_contract COMMAND serial_hotplug_contract_test)
  set_tests_properties(serial_hotplug_contract PROPERTIES TIMEOUT 30)

  # Standalone CI also compiles the real UART owners. In the full project io
  # already builds these sources, so do not compile a second copy.
  if(NOT TARGET io)
    add_library(serial_runtime_compile OBJECT
      ${VISION_SOURCE_DIR}/io/gimbal/gimbal.cpp
      ${VISION_SOURCE_DIR}/io/dm_imu/dm_imu.cpp
      ${VISION_SOURCE_DIR}/io/nav_serial.cpp)
    target_include_directories(serial_runtime_compile PRIVATE ${VISION_SOURCE_DIR})
    target_link_libraries(serial_runtime_compile PRIVATE
      serial Eigen3::Eigen yaml-cpp fmt::fmt spdlog::spdlog nlohmann_json::nlohmann_json)
  endif()
endif()

# Existing contracts use assert(); keep those checks active in Release too.
foreach(name armor_geometry_contract_test nav_protocol_contract_test serial_hotplug_contract_test)
  if(TARGET ${name})
    if(MSVC)
      target_compile_options(${name} PRIVATE /UNDEBUG)
    else()
      target_compile_options(${name} PRIVATE -UNDEBUG)
    endif()
  endif()
endforeach()
