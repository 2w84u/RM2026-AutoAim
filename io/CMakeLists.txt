cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

# 创建目标 io
add_library(io STATIC 
    hikrobot/hikrobot.cpp    
    mindvision/mindvision.cpp  
    usbcamera/usbcamera.cpp  
    camera.cpp
    dm_imu/dm_imu.cpp
    gimbal/gimbal.cpp
    nav_serial.cpp
)

# hikrobot
target_include_directories(io PUBLIC hikrobot/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# mindvision
target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# Legacy C-board CAN is not part of the 2027 production runtime.
if(ENABLE_LEGACY_CBOARD)
  target_sources(io PRIVATE cboard.cpp cboard_uart.cpp)
endif()

target_link_libraries(io PUBLIC MvCameraControl MVSDK usb-1.0 yaml-cpp serial)

if(SPVISION_WITH_ROS2)
  # 添加 ROS2 相关的文件
  target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp ros2/cmd_vel_subscriber.cpp)

  # 设置 ROS2 依赖
  include_directories(${std_msgs_INCLUDE_DIRS})
  target_link_libraries(io PUBLIC
    ${geometry_msgs_TARGETS}
    ${sp_msgs_TARGETS}
    ${std_msgs_TARGETS}
    rclcpp::rclcpp
  )
endif()
