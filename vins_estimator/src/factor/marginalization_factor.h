#pragma once

#include <cstdlib>
#include <unordered_map>

#include <ceres/ceres.h>

#include "utils/tic_toc.h"
#include "utils/utility.h"

constexpr int NUM_THREADS{4};

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    int localSize(int size) const
    {
        return size == 7 ? 6 : size;
    }

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;
    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
};

class MarginalizationInfo
{
public:
    ~MarginalizationInfo();

    int localSize(int size) const;
    int globalSize(int size) const;

    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;
    int m, n;
    std::unordered_map<long, int> parameter_block_size; // global size
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; // local size
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size; // global size
    std::vector<int> keep_block_idx;  // local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps{1e-8};
};

class MarginalizationFactor : public ceres::CostFunction
{
public:
    explicit MarginalizationFactor(MarginalizationInfo *_marginalization_info);

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    MarginalizationInfo *marginalization_info;
};
