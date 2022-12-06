#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double firstImgTime;
double lastImgTime{0.};
int pubCount{1};
bool isFirstImg{true};
bool init_pub{false};

void imgCallback(const sensor_msgs::ImageConstPtr &imgMsg)
{
    const auto currentImgTime = imgMsg->header.stamp.toSec();

    // no operation on first image
    if (isFirstImg)
    {
        isFirstImg = false;
        firstImgTime = currentImgTime;
        lastImgTime = currentImgTime;
        return;
    }

    // check if incoming image stream is stable
    if (currentImgTime < lastImgTime || currentImgTime - lastImgTime > 1.0)
    {
        ROS_WARN("Image stream is not in order! Reset feature tracker...");
        isFirstImg = true;
        lastImgTime = 0;
        pubCount = 1;

        std_msgs::Bool restartMsg;
        restartMsg.data = true;
        pub_restart.publish(restartMsg);

        return;
    }

    lastImgTime = currentImgTime;

    // publish frequency control
    const double pubFreq = 1.0 * pubCount / (currentImgTime - firstImgTime);
    if (std::round(pubFreq) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (std::abs(pubFreq - FREQ) < 0.01 * FREQ)
        {
            firstImgTime = currentImgTime;
            pubCount = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    // retrive image data from message
    cv_bridge::CvImageConstPtr ptr; // TODO: change name
    if (imgMsg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = imgMsg->header;
        img.height = imgMsg->height;
        img.width = imgMsg->width;
        img.is_bigendian = imgMsg->is_bigendian;
        img.step = imgMsg->step;
        img.data = imgMsg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
    {
        ptr = cv_bridge::toCvCopy(imgMsg, sensor_msgs::image_encodings::MONO8);
    }

    cv::Mat show_img = ptr->image;

    TicToc featureTrackerTimer;
    for (int i{0}; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("Processing camera %d", i);

        // mono
        if (i != 1 || !STEREO_TRACK)
        {
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), currentImgTime);
        }
        else
        {
            if (EQUALIZE)
            {
                auto clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
            {
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
            }
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    while (true)
    {
        bool completed{false};
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);

        if (!completed)
            break;
    }

    if (PUB_THIS_FRAME)
    {
        pubCount++;

        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        feature_points->header = imgMsg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            const auto &un_pts = trackerData[i].cur_un_pts;
            const auto &cur_pts = trackerData[i].cur_pts;
            const auto &ids = trackerData[i].ids;
            const auto &pts_velocity = trackerData[i].pts_velocity;
            for (size_t j{0}; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);

                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;
                    feature_points->points.push_back(p);

                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }

        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
        {
            pub_img.publish(feature_points);
        }

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            // cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    // draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    // char name[10];
                    // sprintf(name, "%d", trackerData[i].ids[j]);
                    // cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            // cv::imshow("vis", stereo_img);
            // cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }

    ROS_INFO("Whole feature tracker processing costs: %f", featureTrackerTimer.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if (FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if (!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
            {
                ROS_INFO("load mask success");
            }
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, imgCallback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img", 1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart", 1000);

    ros::spin();

    return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?