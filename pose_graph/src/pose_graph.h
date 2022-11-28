#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <stdio.h>
#include <ros/ros.h>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph
{
public:
	PoseGraph();
	~PoseGraph();

	void loadVocabulary(std::string voc_path);

	void addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);
	void loadKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);
	KeyFrame *getKeyFrame(int index);

	void updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1> &_loop_info);

	// move out
	void registerPub(ros::NodeHandle &n);
	void publish();

	void savePoseGraph();
	void loadPoseGraph();

	nav_msgs::Path path[10];
	nav_msgs::Path base_path;
	CameraPoseVisualization *posegraph_visualization;
	Vector3d t_drift;
	double yaw_drift;
	Matrix3d r_drift;
	// world frame( base sequence or first sequence)<----> cur sequence frame
	Vector3d w_t_vio;
	Matrix3d w_r_vio;

private:
	int detectLoop(KeyFrame *keyframe, int frame_index);
	void addKeyFrameIntoVoc(KeyFrame *keyframe);
	void optimize4DoF();
	void updatePath();

	std::list<KeyFrame *> keyframelist;
	std::mutex m_keyframelist;
	std::mutex m_optimize_buf;
	std::mutex m_path;
	std::mutex m_drift;
	std::thread t_optimization;
	std::queue<int> optimize_buf;

	int global_index;
	int sequence_cnt;
	std::vector<bool> sequence_loop;
	std::map<int, cv::Mat> image_pool;
	int earliest_loop_index;
	int base_sequence;

	BriefDatabase db;
	BriefVocabulary *voc;

	ros::Publisher pub_pg_path;
	ros::Publisher pub_base_path;
	ros::Publisher pub_pose_graph;
	ros::Publisher pub_path[10];
};
