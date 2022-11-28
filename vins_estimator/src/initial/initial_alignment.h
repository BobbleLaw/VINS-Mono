#pragma once

#include <iostream>
#include <map>

#include <Eigen/Dense>

#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include "../feature_manager.h"

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : points{_points}, t{_t} {};

    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    IntegrationBase *pre_integration;
    bool is_key_frame{false};
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);