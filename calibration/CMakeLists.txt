set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

foreach(name calibrate_camera calibrate_handeye calibrate_robotworld_handeye split_video)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
endforeach()

# The parent directory is EXCLUDE_FROM_ALL: capture is still an explicit target.
add_executable(capture capture.cpp)
target_link_libraries(capture PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
