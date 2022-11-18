#pragma once

#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"

using namespace DVision;

class KeyFrame
{
public:
	KeyFrame(double timeStamp, int index, const Eigen::Vector3d &vio_T_w_i, const Eigen::Matrix3d &vio_R_w_i, const cv::Mat &image,
			 const std::vector<cv::Point3f> &point3d, const std::vector<cv::Point2f> &point_2d_uv, const std::vector<cv::Point2f> &point_2d_normal,
			 const std::vector<double> &_point_id, int sequence);
	KeyFrame(double timeStamp, int index, const Eigen::Vector3d &vio_T_w_i, const Eigen::Matrix3d &vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 const cv::Mat &_image, int _loop_index, const Eigen::Matrix<double, 8, 1> &_loop_info,
			 const std::vector<cv::KeyPoint> &keypoints, const std::vector<cv::KeyPoint> &_keypoints_norm, const std::vector<BRIEF::bitset> &_brief_descriptors);

	bool findConnection(KeyFrame *old_kf);

	void computeBRIEFPoint();
	void computeWindowBRIEFPoint();

	bool searchInArea(const BRIEF::bitset window_descriptor,
					  const std::vector<BRIEF::bitset> &descriptors_old,
					  const std::vector<cv::KeyPoint> &keypoints_old,
					  const std::vector<cv::KeyPoint> &keypoints_old_norm,
					  cv::Point2f &best_match,
					  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
						  std::vector<uchar> &status,
						  const std::vector<BRIEF::bitset> &descriptors_old,
						  const std::vector<cv::KeyPoint> &keypoints_old,
						  const std::vector<cv::KeyPoint> &keypoints_old_norm);

	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
								const std::vector<cv::Point2f> &matched_2d_old_norm,
								vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
				   const std::vector<cv::Point3f> &matched_3d,
				   std::vector<uchar> &status,
				   Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);

	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;

	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info);

	Eigen::Vector3d getLoopRelativeT() const;
	Eigen::Quaterniond getLoopRelativeQ() const;
	double getLoopRelativeYaw() const;

	cv::Mat image;
	cv::Mat thumbnail;
	Eigen::Vector3d vio_T_w_i;
	Eigen::Matrix3d vio_R_w_i;
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;
	Eigen::Matrix3d origin_vio_R;
	std::vector<cv::Point3f> point_3d;
	std::vector<cv::Point2f> point_2d_uv;
	std::vector<cv::Point2f> point_2d_norm;
	std::vector<double> point_id;
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::KeyPoint> keypoints_norm;
	std::vector<cv::KeyPoint> window_keypoints;
	std::vector<BRIEF::bitset> brief_descriptors;
	std::vector<BRIEF::bitset> window_brief_descriptors;
	Eigen::Matrix<double, 8, 1> loop_info;
	double time_stamp;
	int sequence;
	int index;
	int local_index;
	int loop_index;
	bool has_fast_point;
	bool has_loop;
};
