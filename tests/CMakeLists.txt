function(add_manual_test name source)
  add_executable(${name} EXCLUDE_FROM_ALL ${source})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
  target_link_libraries(${name} PRIVATE
    ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io ${ARGN})
endfunction()

add_manual_test(camera_test vision/camera_test.cpp)
add_manual_test(detector_verify vision/detector_verify.cpp auto_aim)
add_manual_test(auto_aim_test vision/auto_aim_test.cpp auto_aim)
add_manual_test(auto_buff_test vision/auto_buff_test.cpp auto_buff)

add_manual_test(gimbal_test hardware/gimbal_test.cpp)
add_manual_test(gimbal_feedback_test hardware/gimbal_feedback_test.cpp)
add_manual_test(dm_test hardware/dm_test.cpp)

add_manual_test(planner_test planning/planner_test.cpp auto_aim)
add_manual_test(planner_test_offline planning/planner_test_offline.cpp auto_aim)

if(SPVISION_WITH_ROS2)
  add_manual_test(topic_loop_test ros2/topic_loop_test.cpp)
  add_manual_test(cmd_vel_publisher_test ros2/cmd_vel_publisher_test.cpp)
  add_manual_test(cmd_vel_subscriber_test ros2/cmd_vel_subscriber_test.cpp)
endif()
