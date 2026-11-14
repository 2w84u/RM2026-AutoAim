set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

foreach(name calibrate_camera calibrate_handeye calibrate_robotworld_handeye split_video)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
endforeach()

# capture still uses the old C-board transport; it is never a default target.
if(ENABLE_LEGACY_CBOARD)
  add_executable(capture capture.cpp)
  target_link_libraries(capture PRIVATE ${OpenCV_LIBS} fmt::fmt tools io)
endif()
