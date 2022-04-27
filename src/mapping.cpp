#include "ros/ros.h"

#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "depth_mapping.h"

namespace depthai_ros
{
    /**
     * @brief For depth_mapping_node class
     */

    void depth_mapping_node::init(double ll, double ul, 
        double sub_fact, int mk, float sdm)
    {
        depth_lower_limit = ll * 1.5;
        depth_upper_limit = ul / 1.5;
        subsample_factor = sub_fact;
        std_dev_mul = sdm;
        mean_k = mk;
    }

    /**
     * @brief get_pcl_pointcloud
     * From https://github.com/InverseProject/pose-landmark-graph-slam
     */
    void depth_mapping_node::calc_pcl_pointcloud(cv::Mat depthmap)
    {
        ros::Time start_pcl = ros::Time::now();
        Eigen::Matrix3Xf points_3d = inverse_project_depthmap_into_3d(depthmap);

        ROS_INFO("4. Inverse_project_depthmap_into_3d succeeded");
        int valid_points_cnt = 0;
        ROS_INFO("points_3d Size %lu", points_3d.cols());
        int total_points = points_3d.cols();

        pcl::PointCloud<pcl::PointXYZ>::Ptr empty(new pcl::PointCloud<pcl::PointXYZ>);
        local_pc = empty;

        // reserve memory space for optimization
        local_pc->reserve(total_points);

        // Open CV is in RDF frame
        // ROS is in FLU frame
        for (int i = 0; i < total_points; i += subsample_factor)
        {
            if (points_3d(2, i) < (float)depth_upper_limit && 
                points_3d(2, i) > (float)depth_lower_limit )
            {
                local_pc->push_back(pcl::PointXYZ(
                    points_3d(2, i), -points_3d(0, i), -points_3d(1, i)));
                valid_points_cnt++;
            }
        }

        local_pc->width = valid_points_cnt;
        local_pc->height = 1;
        local_pc->is_dense = true;
        local_pc->resize(valid_points_cnt);

        // https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html
        // The number of neighbors to analyze for each point is set to 50,
        // and the standard deviation multiplier to 1. 
        // What this means is that all points who have a distance larger 
        // than 1 standard deviation of the mean distance to the query point 
        // will be marked as outliers and removed
        
        local_pc = apply_statistical_outlier_removal_filtering(
            mean_k, std_dev_mul, local_pc);

        ROS_INFO("Pointcloud Size %lu", local_pc->points.size());
        ROS_INFO("5. Pointcloud completed %lf\n", (ros::Time::now() - start_pcl).toSec());
    }

    /**
     * @brief inverse_project_depthmap_into_3d
     * From https://github.com/InverseProject/pose-landmark-graph-slam
     */
    Eigen::Matrix3Xf depth_mapping_node::inverse_project_depthmap_into_3d(cv::Mat depthmap)
    {
        // depthmap's size of column (width), row (height), and total number of pixels
        int depthmap_col = depthmap.cols;
        int depthmap_row = depthmap.rows;
        ROS_INFO("cols() %d rows() %d", depthmap_col, depthmap_row);
        int total_pixels = depthmap_col * depthmap_row;

        // creating column index vector : resulting in a row vector (1,total_pixels)
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> col_idx_flat_row_vec =
            Eigen::RowVectorXf::LinSpaced(depthmap_col, 0, depthmap_col - 1).replicate(depthmap_row, 1);
        col_idx_flat_row_vec.resize(1, total_pixels);

        ROS_INFO("1. Created col_idx_flat_row_vec");

        // creating row index vector : resulting in a row vector (1,total_pixels)
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_idx_flat_row_vec =
            Eigen::VectorXf::LinSpaced(depthmap_row, 0, depthmap_row - 1).replicate(1, depthmap_col);
        row_idx_flat_row_vec.resize(1, total_pixels);

        ROS_INFO("2. Created row_idx_flat_row_vec");

        // creating row matrix filled with ones : resulting in a row vector (1,total_pixels)
        auto one_flat_row_vec = Eigen::MatrixXf::Ones(1, total_pixels);

        // getting depth value inside a 2D depth map as a row vector (1,total_pixels)
        Eigen::MatrixXf depth_flat_row_vec = convert_depthmap_to_eigen_row_matrix(depthmap);
        ROS_INFO("3. convert_depthmap_to_eigen_row_matrix succeeded");

        Eigen::Matrix3Xf points(3, total_pixels);
        points.row(0) = col_idx_flat_row_vec;
        points.row(1) = row_idx_flat_row_vec;
        points.row(2) = one_flat_row_vec;


        // https://medium.com/the-inverse-project/opencv-spatial-ai-competition-progress-journal-part-i-ef1ad85016a1
        return intrinsics_.inverse() * points * depth_flat_row_vec.asDiagonal();
    }

    /**
     * @brief convert_depthmap_to_eigen_row_matrix
     * From https://github.com/InverseProject/pose-landmark-graph-slam
     */
    Eigen::MatrixXf depth_mapping_node::convert_depthmap_to_eigen_row_matrix(cv::Mat depthmap)
    {
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mtrx;
        cv::cv2eigen(depthmap, mtrx);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_mtrx = mtrx;

        row_mtrx.resize(1, depthmap.rows * depthmap.cols);

        return row_mtrx;
    }

    /**
     * @brief apply_statistical_outlier_removal_filtering
     * From https://github.com/InverseProject/pose-landmark-graph-slam
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr depth_mapping_node::apply_statistical_outlier_removal_filtering(
        int mean_k_value, float std_dev_multiplier,
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(pointcloud);
        sor.setMeanK(mean_k_value);
        sor.setStddevMulThresh(std_dev_multiplier);
        sor.filter(*output);

        return output;
    }

}