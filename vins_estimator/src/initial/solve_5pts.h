#pragma once

#include <vector>

#include <Eigen/Dense>

class MotionEstimator
{
public:
  using Correspondence = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
  bool solveRelativeRT(const std::vector<Correspondence> &corres, Eigen::Matrix3d &R, Eigen::Vector3d &t);
};
