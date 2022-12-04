#include "initial_sfm.h"

namespace
{
	// refer to Slambook Section 7.7
	bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i,
						 const std::vector<SFMFeature> &sfm_f)
	{
		const int featureCnt = sfm_f.size();
		std::vector<cv::Point2f> pts_2_vector;
		std::vector<cv::Point3f> pts_3_vector;
		pts_2_vector.reserve(featureCnt);
		pts_3_vector.reserve(featureCnt);
		for (int j = 0; j < sfm_f.size(); j++)
		{
			if (!sfm_f[j].state)
				continue;

			const auto &observations = sfm_f[j].observation;
			const auto found = std::find_if(observations.begin(), observations.end(),
											[&i](const auto &observation)
											{
												return observation.first == i;
											});
			if (found != observations.end())
			{
				const auto &img_pts = found->second;
				pts_2_vector.emplace_back(img_pts(0), img_pts(1));
				pts_3_vector.emplace_back(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
			}
		}

		if (pts_2_vector.size() < 15)
		{
			printf("Unstable features tracking, please slowly move you device!\n");
			if (pts_2_vector.size() < 10)
				return false;
		}

		cv::Mat r, rvec, t, D, tmp_r;
		cv::eigen2cv(R_initial, tmp_r);
		cv::Rodrigues(tmp_r, rvec);
		cv::eigen2cv(P_initial, t);
		cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		const bool pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
		if (!pnp_succ)
		{
			return false;
		}

		cv::Rodrigues(rvec, r);
		MatrixXd R_pnp;
		cv::cv2eigen(r, R_pnp);
		MatrixXd T_pnp;
		cv::cv2eigen(t, T_pnp);
		R_initial = R_pnp;
		P_initial = T_pnp;
		return true;
	}

	// refer to Slambook Section 7.5
	void triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
						  const Eigen::Vector2d &point0, const Eigen::Vector2d &point1, Eigen::Vector3d &point3D)
	{
		Eigen::Matrix4d design_matrix = Matrix4d::Zero();
		design_matrix.row(0) = point0[0] * pose0.row(2) - pose0.row(0);
		design_matrix.row(1) = point0[1] * pose0.row(2) - pose0.row(1);
		design_matrix.row(2) = point1[0] * pose1.row(2) - pose1.row(0);
		design_matrix.row(3) = point1[1] * pose1.row(2) - pose1.row(1);

		const Eigen::Vector4d res = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
		point3D(0) = res(0) / res(3);
		point3D(1) = res(1) / res(3);
		point3D(2) = res(2) / res(3);
	}

	void triangulateTwoFrames(int frame0, const Eigen::Matrix<double, 3, 4> &pose0,
							  int frame1, const Eigen::Matrix<double, 3, 4> &pose1,
							  vector<SFMFeature> &sfm_f)
	{
		assert(frame0 != frame1);

		for (int j{0}; j < sfm_f.size(); ++j)
		{
			if (sfm_f[j].state)
				continue;

			bool has0{false}, has1{false};
			Eigen::Vector2d point0;
			Eigen::Vector2d point1;
			for (size_t k{0}; k < sfm_f[j].observation.size(); ++k)
			{
				if (sfm_f[j].observation[k].first == frame0)
				{
					point0 = sfm_f[j].observation[k].second;
					has0 = true;
				}
				if (sfm_f[j].observation[k].first == frame1)
				{
					point1 = sfm_f[j].observation[k].second;
					has1 = true;
				}
			}

			if (has0 && has1)
			{
				Eigen::Vector3d point3D;
				triangulatePoint(pose0, pose1, point0, point1, point3D);
				sfm_f[j].state = true;
				sfm_f[j].position[0] = point3D(0);
				sfm_f[j].position[1] = point3D(1);
				sfm_f[j].position[2] = point3D(2);
			}
		}
	}
}

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		: observed_u(observed_u), observed_v(observed_v)
	{
	}

	static ceres::CostFunction *Create(const double observed_x,
									   const double observed_y)
	{
		return (new ceres::AutoDiffCostFunction<
				ReprojectionError3D, 2, 4, 3, 3>(
			new ReprojectionError3D(observed_x, observed_y)));
	}

	template <typename T>
	bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0];
		p[1] += camera_T[1];
		p[2] += camera_T[2];
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];
		residuals[0] = xp - T(observed_u);
		residuals[1] = yp - T(observed_v);
		return true;
	}

	double observed_u;
	double observed_v;
};

GlobalSFM::GlobalSFM() {}

// 	q w_R_cam t w_R_cam
//  c_rotation cam_R_w
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Eigen::Quaterniond *q, Eigen::Vector3d *T, int l,
						  const Eigen::Matrix3d &relative_R, const Eigen::Vector3d &relative_T,
						  std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points)
{
	const int feature_num = sfm_f.size();

	// initialize reference frame and current frame
	q[l] = Eigen::Quaterniond::Identity();
	T[l].setZero();

	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;

	// the transform from reference frame to every other frames
	Eigen::Matrix3d c_Rotation[frame_num];
	Eigen::Vector3d c_Translation[frame_num];
	Eigen::Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	// initialize reference frame and current frame transform
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -(c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

	// 1: trangulate between reference to current frame
	// 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
	for (int i{l}; i < frame_num - 1; i++)
	{
		// solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;

			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	// 3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

	// 4: solve pnp l-1; triangulate l-1 ----- l
	//              l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;

		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];

		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}

	// 5: triangulate all other points
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state)
			continue;

		if (sfm_f[j].observation.size() >= 2)
		{
			const int frame0 = sfm_f[j].observation[0].first;
			const auto &point0 = sfm_f[j].observation[0].second;
			const int frame1 = sfm_f[j].observation.back().first;
			const auto &point1 = sfm_f[j].observation.back().second;
			Eigen::Vector3d point3D;
			triangulatePoint(Pose[frame0], Pose[frame1], point0, point1, point3D);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point3D(0);
			sfm_f[j].position[1] = point3D(1);
			sfm_f[j].position[2] = point3D(2);
		}
	}

	// Full BA
	ceres::Problem problem;
	auto *local_parameterization = new ceres::QuaternionParameterization();
	for (int i = 0; i < frame_num; i++)
	{
		// find optimization ONLY for poses, not landmarks
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);

		// fixed reference frame and current frame
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)
	{
		if (!sfm_f[i].state)
			continue;

		for (size_t j{0}; j < sfm_f[i].observation.size(); j++)
		{
			int l = sfm_f[i].observation[j].first;
			auto *cost_function = ReprojectionError3D::Create(sfm_f[i].observation[j].second.x(),
															  sfm_f[i].observation[j].second.y());

			problem.AddResidualBlock(cost_function, nullptr, c_rotation[l], c_translation[l],
									 sfm_f[i].position);
		}
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if (summary.termination_type != ceres::CONVERGENCE && summary.final_cost >= 5e-03)
	{
		return false
	}

	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0];
		q[i].x() = c_rotation[i][1];
		q[i].y() = c_rotation[i][2];
		q[i].z() = c_rotation[i][3];
		q[i] = q[i].inverse();
	}

	for (int i = 0; i < frame_num; i++)
	{
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
	}

	for (int i = 0; i < sfm_f.size(); i++)
	{
		if (sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}

	return true;
}
