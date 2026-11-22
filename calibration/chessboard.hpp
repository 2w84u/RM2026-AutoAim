#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace calibration
{
// pattern_size counts inner corners, not black/white squares.
inline bool find_chessboard_corners(
  const cv::Mat & image, const cv::Size & pattern_size, std::vector<cv::Point2f> & corners)
{
  cv::Mat gray;
  if (image.channels() == 1)
    gray = image;
  else
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  if (!cv::findChessboardCorners(
        gray, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE))
    return false;

  cv::cornerSubPix(
    gray, corners, {5, 5}, {-1, -1},
    {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001});
  return true;
}
}  // namespace calibration
