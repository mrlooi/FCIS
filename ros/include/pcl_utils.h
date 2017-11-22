//
// Created by vincentlv on 17-3-16.
//

#ifndef VISION_PCL_UTILS_H
#define VISION_PCL_UTILS_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

template <typename PointT>
std::vector<int> filterCloudNan(const pcl::PointCloud<PointT>& cloud, pcl::PointCloud<PointT>& out_cloud)
{
    const int cloud_sz = cloud.points.size();

    auto& out_pts = out_cloud.points;
    out_pts.clear();
    out_pts.reserve(cloud_sz);

    std::vector<int> good_indices;

    for (unsigned int i = 0; i < cloud_sz; i++)
    {
        const PointT& pt = cloud.points[i];

        if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z))
        {
            out_pts.push_back(pt);
            good_indices.push_back(i);
        }
    }
    return good_indices;
}

inline void get_outliers(std::vector<int>& outliers, pcl::PointIndices::Ptr inliers, const int cloud_sz)
{
    const int inliers_sz = inliers->indices.size();
    int inlier_counter = 0;
    for (int i = 0; i < inliers_sz; ++i)
    {   
        int _ = inliers->indices[i];
        for (int j = inlier_counter; j < _; ++j)
        {
            outliers.push_back(j);
        }
        inlier_counter = _ + 1; 
    }
    for (int i = inlier_counter; i < cloud_sz; ++i)
    {
        outliers.push_back(i);
    }
}

template <typename PointT>
std::vector<PointT> find_pointcloud_pose(const pcl::PointCloud<PointT>& cloud, Eigen::Vector3f& transform, Eigen::Quaternionf& quarternion)
{
    std::vector<PointT> min_max_pts(2);

    find_pointcloud_pose(cloud, transform, quarternion, min_max_pts[0], min_max_pts[1]);

	return min_max_pts;
}

template <typename PointT>
void find_pointcloud_pose(const pcl::PointCloud<PointT>& cloud, Eigen::Vector3f& transform, Eigen::Quaternionf& quarternion, PointT& min_pt, PointT& max_pt)
{
// //calculate smallest bounding box
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
//
//        // Transform the original cloud to the origin where the principal components correspond to the axes.
    const Eigen::Matrix4f mat4f_identity = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f projectionTransform(mat4f_identity);
//
    projectionTransform.block(0,0,3,3) = eigenVectorsPCA.transpose();
    projectionTransform.block(0,3,3,1) = -1.f * (projectionTransform.block(0,0,3,3) * pcaCentroid.head(3));

    pcl::PointCloud<PointT> cloudPointsProjected;

    pcl::transformPointCloud(cloud, cloudPointsProjected, projectionTransform);
//
//        // Get the minimum and maximum points of the transformed cloud.
    pcl::getMinMax3D(cloudPointsProjected, min_pt, max_pt);
    const Eigen::Vector3f meanDiagonal = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());
//
//        // Final transform & quarternion
    transform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head(3);
    quarternion = Eigen::Quaternionf(eigenVectorsPCA);
	
}

#endif //VISION_PCL_UTILS_H
