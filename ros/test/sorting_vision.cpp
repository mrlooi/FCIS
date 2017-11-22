#include <ros/ros.h>
#include <ros/package.h>

#include <iostream>

// pcl ros messages and pointcloud2
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>

// pcl core
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
// pcl vis
#include <pcl/visualization/cloud_viewer.h>

// // pcl io
// #include <pcl/io/pcd_io.h>

// pcl ransac
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

// opencv
#include <opencv2/opencv.hpp>

// ros pose msg
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>

// ros marker vis
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// custom
#include "vision/BBox.h"
#include "vision/Detection2.h"

#include "pcl_utils.h"
#include "geometry_msg_utils.h"

using namespace std;


#define BASE_LINK "base_link"
#define CAMERA_DEPTH_TOPIC "/kinect2/qhd/points"
#define CAMERA_FRAME "kinect2_rgb_optical_frame"
#define CAMERA_COLOR_TOPIC "/kinect2/hd/image_color"
#define CAMERA_FRAME_COLOR "kinect2_rgb_optical_frame"
#define FIND_OBJECT_TOPIC "/kinect2/hd/mask_service_box_envelope"
#define OBJECT_POSE_TOPIC "object_pose"
#define MARKER_TOPIC "object_marker"

#define VISUALIZE
#ifdef VISUALIZE
pcl::visualization::CloudViewer viewer("Cloud Viewer");
#endif

typedef pcl::PointXYZRGB PointT;

class Vision
{
public:
    Vision(ros::NodeHandle& nh): nh_(nh)
    {
        initDataBuffer();
        initRansac();

        setup_coms();
    }

    void run()
    {
        ros::Rate r(30);

        while (ros::ok())
        {
          ros::spinOnce();
          r.sleep();
        }

        ROS_WARN("Shutting down...");
        pc_sub_.shutdown();
    }

private:
    void initDataBuffer()
    {
        cloud_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        cloud_raw_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    }


    void initRansac()
    {
        ransac_.setOptimizeCoefficients (true);
        ransac_.setModelType (pcl::SACMODEL_PLANE);
        ransac_.setMethodType (pcl::SAC_RANSAC);
        ransac_.setDistanceThreshold (0.01);
    }

    void setup_coms()
    {
        tf_listener_ = std::unique_ptr<tf::TransformListener>(new tf::TransformListener());

        pc_sub_ = nh_.subscribe(CAMERA_DEPTH_TOPIC, 1, &Vision::cameraDepthCb, this);
        find_object_srv_ = nh_.serviceClient<vision::Detection2>(FIND_OBJECT_TOPIC);
        object_pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>(OBJECT_POSE_TOPIC, 1);
        vis_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(MARKER_TOPIC,1);
    }

    void cameraDepthCb(const sensor_msgs::PointCloud2& msg)
    {
        // tf_listener_->waitForTransform(BASE_LINK, CAMERA_FRAME, ros::Time::now(), ros::Duration(1));
        // pcl_ros::transformPointCloud(BASE_LINK, *msg, transformed_pc2_msg_, *tf_listener_);

        pcl::fromROSMsg<PointT>(msg, *cloud_raw_);

        cloud_ = cloud_raw_;
        // filterCloudNan(*cloud_raw_, *cloud_);
        const int cloud_sz = cloud_->points.size();  // 960*540=518400

        ++counter;

        if (cloud_sz == 0)
        {
        	return;
        }

#ifdef VISUALIZE
        // show cloud
        viewer.showCloud(cloud_);
#endif

        if (find_object())
        {
            ROS_INFO_STREAM("Find object success!");
        }
        else
        {
            ROS_ERROR_STREAM("Find object failed");
        }
    
        return; 
    
    }

    bool find_object()
    {
        vision::Detection2 find_object;
        if (!find_object_srv_.call(find_object))
        {
          // ROS_ERROR_STREAM("find object call failed");
          return false;
        }

        auto& detection = find_object.response.detection;

        // store pose for publishing
        std::vector<geometry_msgs::Pose> pose_array;
        std::vector<geometry_msgs::Vector3> pose_size_array;

        pose_array.reserve(detection.size());
        pose_size_array.reserve(detection.size());

        for (int i = 0; i < detection.size(); i++)
        {
            const auto& det = detection[i];
            if (det.bbox.size() >= 4)
            {
                int xmin, ymin, xmax, ymax;

                xmin = det.bbox[0]/2;
                ymin = det.bbox[1]/2;
                xmax = det.bbox[2]/2;
                ymax = det.bbox[3]/2;

                // ROS_INFO("CLOUD WIDTH %d HEIGHT %d, size: %d", cloud_->width, cloud_->height, (int)cloud_->points.size());

                if (xmin >= 0 && xmax < cloud_->width && xmin < xmax && ymin >= 0 && ymax < cloud_->height && ymin < ymax)
                {
                    // get contours of the detection
                    // contour format: x,y,x,y,...
                    std::vector<cv::Point> contours;
                    const int cnt_length = det.contours.size()/2;
                    contours.reserve(cnt_length);
                    for (int j = 0; j < cnt_length; j++)
                    {
                      contours.push_back(cv::Point(det.contours[j*2] / 2, det.contours[j*2+1] / 2));
                    }

                    // ROS_INFO("Contours size %d", (int)contours.size());

                    geometry_msgs::Pose det_pose; 
                    geometry_msgs::Vector3 det_size;
                    if (!get_det_pose({xmin, ymin, xmax, ymax},contours, det_pose, det_size))
                        continue;

                    pose_array.push_back(det_pose);
                    pose_size_array.push_back(det_size);
                } 
                else
                {
                    ROS_ERROR("INVALID BBOX POINTS %d %d %d %d", xmin, ymin, xmax, ymax);
                    return false;
                }

            }
        }

        geometry_msgs::PoseArray final_pose_array = publish_pose(pose_array);
        publish_pose_size(final_pose_array, pose_size_array);

        return true;
    }

    geometry_msgs::PoseArray publish_pose(const std::vector<geometry_msgs::Pose>& pose_array)
    {
        // tf_listener_->waitForTransform(BASE_LINK, CAMERA_FRAME, ros::Time::now(), ros::Duration(1.0));
        geometry_msgs::PoseArray final_pose_array;
        final_pose_array.header.frame_id = CAMERA_FRAME;

        double offset = -0.1;

        for (int i = 0; i < pose_array.size(); ++i)
        {

            const geometry_msgs::Pose& pose = pose_array[i];

            geometry_msgs::Pose final_pose;

            geometry_msgs::Pose temp_pose = pose_axis_offset(pose, "x", offset);
            if (temp_pose.position.z > pose.position.z)
                final_pose = pose;
            else
                final_pose = pose_axis_rotation(pose, "z", 180);
    
            // geometry_msgs::Pose transformed_pose;
            // tf_listener_->transformPose(BASE_LINK, pose, transformed_pose);
            final_pose_array.poses.push_back(final_pose);
        }
        object_pose_pub_.publish(final_pose_array);

        return final_pose_array;
    }

    void publish_pose_size(const geometry_msgs::PoseArray& pose_array, const std::vector<geometry_msgs::Vector3>& pose_size_array)
    {
        std::vector<geometry_msgs::Pose> pose_vector;
        pose_vector.reserve(pose_array.poses.size());
        for (int i = 0; i < pose_array.poses.size(); ++i)
        {
            pose_vector.push_back(pose_array.poses[i]);
        }

        publish_pose_size(pose_vector, pose_size_array);
    }

    void publish_pose_size(const std::vector<geometry_msgs::Pose>& pose_array, const std::vector<geometry_msgs::Vector3>& pose_size_array)
    {
        visualization_msgs::MarkerArray marker_array;

        for (int i = 0; i < pose_array.size(); ++i)
        {
            const geometry_msgs::Pose& pose = pose_array[i];
            const geometry_msgs::Vector3& pose_size = pose_size_array[i];

            visualization_msgs::Marker model_marker;
            model_marker.header.frame_id = CAMERA_FRAME;
            // model_marker.header.stamp = ros::Time();
            // model_marker.ns = model.str();
            model_marker.id = i;
            model_marker.type = visualization_msgs::Marker::CUBE;
            model_marker.scale.x = pose_size.x;
            model_marker.scale.y = pose_size.y;
            model_marker.scale.z = pose_size.z;

            model_marker.pose = pose;
            model_marker.color.r = 0;
            model_marker.color.g = 0;
            model_marker.color.b = 255;
            model_marker.color.a = 0.5;

            model_marker.action = visualization_msgs::Marker::ADD;

            marker_array.markers.push_back(model_marker);
        }

        vis_pub_.publish(marker_array);
    }


    bool get_det_pose(const std::vector<int>& bbox, const std::vector<cv::Point>& contours, geometry_msgs::Pose& det_pose, geometry_msgs::Vector3& det_size)
    {
        if (bbox.size() <4)
        {
            ROS_ERROR("require a point");
            return false;
        }
        if (cloud_->points.size() == 0)
        {
            ROS_ERROR("cloud_ not registered");
            return false;
        }

        // compute mask over image to limit pointcloud to mask segment
        cv::Mat mask = cv::Mat::zeros(cloud_->height, cloud_->width, CV_8UC1);
        // ROS_INFO("Fill convex poly...");
        // for (int i = 0; i < contours.size(); ++i)
        // {
        //     const auto& cnt = contours[i];
        //     std::cout << cnt.x << " " << cnt.y << "\n"; 
        // }
        cv::fillConvexPoly(mask, contours, cv::Scalar(255));


        // extract pointcloud in mask
        pcl::PointCloud<PointT>::Ptr subcloud(new pcl::PointCloud<PointT>);

        int xmin, ymin, xmax, ymax;

        xmin = bbox[0];
        ymin = bbox[1];
        xmax = bbox[2];
        ymax = bbox[3];

        for (int r = ymin; r < ymax; r += 1) 
        {
            for (int c = xmin; c < xmax; c += 1) 
            {
                if (mask.at<unsigned char>(r,c) == 255)
                {
                    int pos = cloud_->width * r + c;
                    const PointT& pt = cloud_->points.at(pos);
                    if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z))
                        subcloud->points.push_back(pt);
                }
            }
        }

        if (subcloud->size() == 0)
        {
            ROS_ERROR("Empty subcloud! %d %d %d %d", xmin, xmax, ymin, ymax);
            return false;
        }
        // ROS_INFO("Subcloud size %d", (int)subcloud->size());

        // compute RANSAC on mask pointcloud to extract dominant planar points
        pcl::PointCloud<PointT>::Ptr ransac_filtered_subcloud(new pcl::PointCloud<PointT>);
        ransacSegment(subcloud, ransac_filtered_subcloud);

        if (ransac_filtered_subcloud->size() == 0)
        {
            ROS_ERROR("Ransac filter returned empty subcloud! %d %d %d %d", xmin, xmax, ymin, ymax);
            return false;
        }
        // ROS_INFO("Ransac cloud size %d", (int)ransac_filtered_subcloud->size());

        // get the average of all the points to find middle point
        float sumx = 0;
        float sumy = 0;
        float sumz = 0;
        for(int j = 0; j < ransac_filtered_subcloud->points.size(); j++)
        {
            sumx = sumx + ransac_filtered_subcloud->points.at(j).x;
            sumy = sumy + ransac_filtered_subcloud->points.at(j).y;
            sumz = sumz + ransac_filtered_subcloud->points.at(j).z;
        }
        float avgx = sumx/ransac_filtered_subcloud->size();
        float avgy = sumy/ransac_filtered_subcloud->size();
        float avgz = sumz/ransac_filtered_subcloud->size();

        // get pose
        Eigen::Vector3f transform; 
        Eigen::Quaternionf quarternion; 
        PointT min_pt, max_pt;
        find_pointcloud_pose(*ransac_filtered_subcloud, transform, quarternion, min_pt, max_pt);

        det_size.x = max_pt.x - min_pt.x;
        det_size.y = max_pt.y - min_pt.y;
        det_size.z = max_pt.z - min_pt.z;

        det_pose.position.x = avgx;
        det_pose.position.y = avgy;
        det_pose.position.z = avgz;
        det_pose.orientation.x = quarternion.x();
        det_pose.orientation.y = quarternion.y();
        det_pose.orientation.z = quarternion.z();
        det_pose.orientation.w = quarternion.w();

        return true;
    }

    void ransacSegment(const pcl::PointCloud<PointT>::Ptr in_cloud, pcl::PointCloud<PointT>::Ptr resultcloud)
    {
        if (in_cloud->points.size() == 0)
        {
            return;
        }
        ransac_.setInputCloud (in_cloud);

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

        ransac_.segment (*inliers, *coefficients);

        if (inliers->indices.size () == 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    //        return (-1);
        }

        resultcloud->points.resize(inliers->indices.size());
        for (size_t i = 0; i < inliers->indices.size(); ++i) 
        {
            const PointT& pt = in_cloud->points[inliers->indices[i]];
            resultcloud->points[i] = pt;
        }
        resultcloud->width = resultcloud->points.size();
        resultcloud->height = 1;

    }
    
    // cloud data
    pcl::PointCloud<PointT>::Ptr cloud_raw_;
    pcl::PointCloud<PointT>::Ptr cloud_;
    sensor_msgs::PointCloud2 transformed_pc2_msg_;

    // ransac segmentation
    pcl::SACSegmentation<PointT> ransac_;
    pcl::PointCloud<PointT>::Ptr ransac_filtered_cloud_;

    // misc
    int counter = 0;
protected:
    // ros 
    ros::NodeHandle& nh_;
    std::unique_ptr<tf::TransformListener> tf_listener_;

    ros::Subscriber pc_sub_;
    ros::ServiceClient find_object_srv_;
    ros::Publisher object_pose_pub_;
    ros::Publisher vis_pub_;
    
};


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "singulation");

    ros::NodeHandle nh;

    std::unique_ptr<Vision> vision(new Vision(nh));

    vision->run();

    ros::MultiThreadedSpinner spinner(4); // Use 4 threads
    spinner.spin(); // spin() will not return until the node has been shutdown


    ros::shutdown();

	return 0;
}