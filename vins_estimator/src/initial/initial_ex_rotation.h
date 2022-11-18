#pragma once

#include <vector>

#include <Eigen/Dense>

#include "../parameters.h"

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialEXRotation
{
public:
    InitialEXRotation();

    using Correspondence = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
    bool CalibrationExRotation(const std::vector<Correspondence> &corres, const Eigen::Quaterniond &delta_q_imu, Eigen::Matrix3d &calib_ric_result);

private:
    int frame_count;

    std::vector<Eigen::Matrix3d> Rc;
    std::vector<Eigen::Matrix3d> Rimu;
    std::vector<Eigen::Matrix3d> Rc_g;
    Eigen::Matrix3d ric;
};
