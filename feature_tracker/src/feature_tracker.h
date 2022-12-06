#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utils/tic_toc.h"

#include "parameters.h"

bool inBorder(const cv::Point2f &pt);

template <typename T>
void reduceVector(std::vector<T> &vals, const std::vector<uchar> &keep);

class FeatureTracker
{
public:
  FeatureTracker();

  void readIntrinsicParameter(const std::string &filename);

  void readImage(const cv::Mat &img, double timeStamp);

  bool updateID(unsigned int i);

  // for debug
  void showUndistortion(const std::string &imgName);

public:
  cv::Mat mask;
  cv::Mat fisheye_mask;
  cv::Mat prev_img, cur_img, forw_img;
  std::vector<cv::Point2f> n_pts;
  std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
  std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
  std::vector<cv::Point2f> pts_velocity;
  std::vector<int> ids;
  std::vector<int> track_cnt;
  std::map<int, cv::Point2f> cur_un_pts_map;
  std::map<int, cv::Point2f> prev_un_pts_map;
  camodocal::CameraPtr m_camera;
  double cur_time;
  double prev_time;

  static int n_id;

private:
  void rejectWithF();
  void setMaskAroundFeatures();
  void addPoints();
  void undistortedPoints(); // TODO: ... and calculate feature velocity

private:
};
