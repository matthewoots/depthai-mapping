#include <string>
#include <iostream>
#include <memory>
#include <queue>

#include <Eigen/Dense>
#include <math.h>

#include "ros/ros.h"

// include depthai library
#include <depthai/depthai.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


namespace depthai_ros
{
    using CV_mat_ptr = std::shared_ptr<cv::Mat>;

    class depth_mapping_node
    {
        public:

            double depth_lower_limit, depth_upper_limit;
            Eigen::Matrix3f intrinsics_;

            void init(double ll, double ul, 
                double sub_fact, int mk, float sdm);

            void calc_pcl_pointcloud(cv::Mat depthmap);

            pcl::PointCloud<pcl::PointXYZ>::Ptr local_pc;

        private:

            Eigen::Matrix3Xf inverse_project_depthmap_into_3d(cv::Mat depthmap);

            Eigen::MatrixXf convert_depthmap_to_eigen_row_matrix(cv::Mat depthmap);

            pcl::PointCloud<pcl::PointXYZ>::Ptr apply_statistical_outlier_removal_filtering(
                int mean_k_value, float std_dev_multiplier,
                pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud);

            double subsample_factor;
            int mean_k; 
            float std_dev_mul;
            
    };

    /**
     * DepthMapPublisher is a ROS node which launches OAK-D streams as specified in the config file.
     * Currently it is capable of publishing depth map produced by OAK-D.
     * Streaming AI based objects position is WIP
     */
    class depth_publisher_node
    {
        public:

            /** @brief Mutexes */
            std::mutex depth_mutex, odom_mutex;

            /** @brief Odom queue is needed to synchronize odom and pcl at timestamp */
            std::queue<std::pair<double, Eigen::Affine3d>> odom_queue;

            /**
             * @param intrinsics (const Eigen::Matrix3f&) : 3x3 camera intrinsics
             *
             * camera intrinsics (K)
             * |f_u   0   o_u|
             * | 0   f_v  0_v|
             * | 0    0    1 |
             *
             * (f_u, f_v) -> focal length (horizontal, vertical)
             * (o_u, o_v) -> principal point (horizontal, vertical)
             */
            Eigen::Matrix3f intrinsics;

            depth_mapping_node map_node;

            /**
             * Constructor
             */
            depth_publisher_node(
                ros::NodeHandle &nodeHandle);

            /**
             * Destructor
             * Stop the device before stopping the ROS node.
             */
            ~depth_publisher_node();

            void publisher();

            void setup_depth_stream();

            void update_depth_timer(const ros::TimerEvent &);

            void update_map(const ros::TimerEvent &);

            bool device_poll();

            cv::Mat disparity_filter(
                cv::Mat disparity, cv::Mat right, double lambda, double sigma);

            void update_depth_pair(cv::Mat depth_mat)
            {
                std::lock_guard<std::mutex> d_lock(depth_mutex);
                depth_w_stamp.first = ros::Time::now().toSec();
                depth_w_stamp.second = depth_mat;
                return;
            }

            cv::Mat get_depth_pair()
            {
                std::lock_guard<std::mutex> d_lock(depth_mutex);
                cv::Mat new_mat = depth_w_stamp.second;
                return new_mat;
            }

            bool check_depth_mat()
            {
                if(depth_w_stamp.second.empty())
                    return false;
                else
                    return true;
            }

            void sync_odom_depth()
            {
                std::lock_guard<std::mutex> d_lock(depth_mutex);
                std::lock_guard<std::mutex> o_lock(odom_mutex);
                while (!odom_queue.empty()) 
                    odom_queue.pop();
            }

        private:
            ros::NodeHandle _nh;

            ros::Timer update_depth, map_timer;
            ros::Publisher depth_image_pub, disparity_image_pub, local_pcl_pub;

            /** @brief Resolution types defined by DEPTHAI like 800p or 480p */
            std::string _mono_resolution;

            /** @brief Topic of depth message */
            std::string _depth_map_topic;

            /** 
             * @brief DEPTHAI Parameters 
             * @param mono_resolution (std::string): Resolution of the camera object
             * @param depth_map_topic_ (std::string): Topic on which depth map will be broadcasted (etc /depth_map)
             * @param confidence (int): Confidence threshold for disparity calculation (Confidence threshold value 0 to 255)
             * @param lr_check (bool): Computes and combines disparities in both L-R and R-L directions, and combine them
             * @param extended (bool): Disparity range increased from 95 to 190, combined from full resolution and downscaled images. Suitable for short range objects 
             * @param subpixel (bool): Better accuracy for longer distance, fractional disparity 32-levels:
             * 
             */
            int _confidence, _lr_check_thres, _median_kernal_size;
            bool _lr_check, _extended, _subpixel;
            double baseline;
            double hfov, vfov;
        
            double _update_interval;
            double _max_range_clamp, _min_range_clamp;

            /** @brief Poll till we timeout and close the node */
            double empty_poll_count_time = 0.0;

            /** @brief Calculated intrinsic parameters parameters */
            double image_width_in_pixels, image_height_in_pixels, focal_length_in_pixels;

            bool init = false, registered = false;

            /** @brief Subsample skips pixels to use in pcl, downsample shrinks the cv::Mat */
            int _subsample; double _downsample;

            /** @brief Statistical Outlier Removal Filter parameters */
            int _mean_k; double _std_dev_mul;

            /** @brief WLS Disparity filter parameters */
            double _lambda, _sigma;

            ros::Time node_start_time;

            ros::Publisher _depth_map_pub;

            std::shared_ptr<dai::node::StereoDepth> stereo_depth;
            std::shared_ptr<dai::node::XLinkOut> xout_disp;
            std::shared_ptr<dai::node::XLinkOut> xout_right;

            // Define sources and outputs
            std::shared_ptr<dai::node::MonoCamera> mono_left;
            std::shared_ptr<dai::node::MonoCamera> mono_right;

            dai::Pipeline dev_pipeline;
            std::shared_ptr<dai::Device> device;
            std::shared_ptr<dai::DataOutputQueue> disparity_data;
            std::shared_ptr<dai::DataOutputQueue> right_data;

            std::pair<double, cv::Mat> depth_w_stamp;
            std::pair<double, Eigen::Affine3d> odom_tf_w_stamp;
    };
}