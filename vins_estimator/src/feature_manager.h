#pragma once

#include <list>
#include <vector>

#include <Eigen/Dense>

#include "parameters.h"

struct FeaturePerId;

class FeatureManager
{
public:
  explicit FeatureManager(Eigen::Matrix3d _Rs[]);

  void setRic(Eigen::Matrix3d _ric[]);

  void clearState();

  int getFeatureCount();

  bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);

  using Correspondence = std::pair<Eigen::Vector3d, Eigen::Vector3d>;
  std::vector<Correspondence> getCorresponding(int frame_count_l, int frame_count_r);

  void setDepth(const Eigen::VectorXd &x);
  void clearDepth(const Eigen::VectorXd &x);
  VectorXd getDepthVector();

  void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);

  void removeFront(int frame_count);
  void removeBack();
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeFailures();
  void removeOutlier();

  void debugShow();

  std::list<FeaturePerId> feature;
  int last_track_num;

private:
  const Eigen::Matrix3d *Rs;
  Eigen::Matrix3d ric[NUM_OF_CAM];
};
