#pragma once

#include <map>

#include <Eigen/Dense>

struct SFMFeature
{
	bool state; // TODO: change name
	int id;
	vector<pair<int, Vector2d>> observation;
	double position[3];
	double depth;
};

class GlobalSFM
{
public:
	GlobalSFM();

	bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
				   const Eigen::Matrix3d &relative_R, const Eigen::Vector3d &relative_T,
				   std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points) const;
};
