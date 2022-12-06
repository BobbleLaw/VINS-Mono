#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    constexpr int kImgBorderSize{1};
    const int x = cvRound(pt.x);
    const int y = cvRound(pt.y);
    return kImgBorderSize <= x && x < COL - kImgBorderSize && kImgBorderSize <= y && y < ROW - kImgBorderSize;
}

template <typename T>
void reduceVector(std::vector<T> &vals, const std::vector<uchar> &keep)
{
    int size{0};
    for (size_t i = {0}; i < int(vals.size()); i++)
        if (keep[i])
            vals[size++] = vals[i];
    vals.resize(size);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::readIntrinsicParameter(const std::string &filename)
{
    ROS_INFO("Reading parameters from %s", filename.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(filename);
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cur_time = _cur_time;
    cv::Mat img;
    if (EQUALIZE)
    {
        const auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc eqTimer;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %f ms", eqTimer.toc());
    }
    else
    {
        img = _img;
    }

    if (forw_img.empty())
    {
        prev_img = img;
        cur_img = img;
    }
    forw_img = img;

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (size_t i{0}; i < forw_pts.size(); i++)
            status[i] = status[i] && inBorder(forw_pts[i]);

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);

        ROS_DEBUG("Temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &cnt : track_cnt)
    {
        cnt++;
    }

    if (PUB_THIS_FRAME)
    {
        rejectWithF();

        ROS_DEBUG("Set mask begins");
        TicToc t_m;
        setMaskAroundFeatures();
        ROS_DEBUG("Set mask costs %f ms", t_m.toc());

        ROS_DEBUG("Detect feature begins");
        TicToc t_t;
        const int moreFeatureCnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (moreFeatureCnt > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            cv::goodFeaturesToTrack(forw_img, n_pts, moreFeatureCnt, 0.01, MIN_DIST, mask);
        }
        else
        {
            n_pts.clear();
        }
        ROS_DEBUG("Detect feature costs: %f ms", t_t.toc());

        ROS_DEBUG("Add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("Select feature costs: %f ms", t_a.toc());
    }

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }

    return false;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }

    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;

        std::vector<cv::Point2f> un_cur_pts(cur_pts.size());
        std::vector<cv::Point2f> un_forw_pts(forw_pts.size());
        for (size_t i{0}; i < cur_pts.size(); i++)
        {
            auto toNormalizedCoord = [&m_camera](const cv::Point2f &pt) -> cv::Point2f
            {
                Eigen::Vector3d pt3D;
                m_camera->liftProjective({pt.x, pt.y}, pt3D);
                pt3D.x() = FOCAL_LENGTH * pt3D.x() / pt3D.z() + COL / 2.0;
                pt3D.y() = FOCAL_LENGTH * pt3D.y() / pt3D.z() + ROW / 2.0;
                return {pt3D.x(), pt3D.y()};
            };

            un_cur_pts[i] = toNormalizedCoord(cur_pts[i]);
            un_forw_pts[i] = toNormalizedCoord(forw_pts[i]);
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);

        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::setMaskAroundFeatures()
{
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    struct Feature
    {
        Feature(const cv::Point2f &pt, int id, int times) : pt(pt), id(id), trackedTimes(times) {}
        cv::Point2f pt;
        int id;
        int trackedTimes;
    };

    std::vector<Feature> features;
    features.reserve(forw_pts.size());
    for (size_t i{0}; i < forw_pts.size(); ++i)
    {
        features.emplace_back(forw_pts[i], ids[i], track_cnt[i]);
    }

    // prefer to keep features that are tracked for long time
    sort(features.begin(), features.end(), [](const auto &a, const auto &b)
         { return a.trackedTimes > b.trackedTimes; });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (const auto &feature : features)
    {
        if (mask.at<uchar>(feature.pt) == 255)
        {
            forw_pts.push_back(feature.pt);
            ids.push_back(feature.id);
            track_cnt.push_back(feature.trackedTimes);
            cv::circle(mask, feature.pt, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (const auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();

    for (size_t i{0}; i < cur_pts.size(); i++)
    {
        Eigen::Vector3d pt3D;
        m_camera->liftProjective({cur_pts[i].x, cur_pts[i].y}, pt3D);
        const cv::Point2f pt2D{pt3D.x() / pt3D.z(), pt3D.y() / pt3D.z()};
        cur_un_pts.push_back(pt2D);
        cur_un_pts_map.insert({ids[i], pt2D});
    }

    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        const double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (size_t i{0}; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                const auto found = prev_un_pts_map.find(ids[i]);
                if (found != prev_un_pts_map.end())
                {
                    pts_velocity.push_back((cur_un_pts[i] - it->second) / dt);
                }
                else
                {
                    pts_velocity.push_back({0., 0.});
                }
            }
            else
            {
                pts_velocity.push_back({0., 0.});
            }
        }
    }
    else
    {
        for (size_t i{0}; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back({0., 0.});
        }
    }

    prev_un_pts_map = cur_un_pts_map;
}
