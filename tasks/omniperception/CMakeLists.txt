cmake_minimum_required(VERSION 3.16)

# Perceptron only consumes auto_aim::YOLO, so this module does not need the
# inference runtime directly (OpenVINO or OpenCV DNN both live in auto_aim).
add_library(omniperception OBJECT 
    decider.cpp
    perceptron.cpp
)