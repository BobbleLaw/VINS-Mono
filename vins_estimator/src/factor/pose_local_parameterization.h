#pragma once

#include <ceres/ceres.h>

// TODO: use ceres::Manifold
class PoseLocalParameterization : public ceres::LocalParameterization
{
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
    bool ComputeJacobian(const double *x, double *jacobian) const override;

    int GlobalSize() const override { return 7; };
    int LocalSize() const override { return 6; };
};
