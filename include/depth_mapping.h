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

            void init(double ll, double ul, double sub_fact);

            void calc_pcl_pointcloud(cv::Mat depthmap);

            pcl::PointCloud<pcl::PointXYZ>::Ptr local_pc;

        private:

            Eigen::Matrix3Xf inverse_project_depthmap_into_3d(cv::Mat depthmap);

            Eigen::MatrixXf convert_depthmap_to_eigen_row_matrix(cv::Mat depthmap);

            double subsample_factor;
            
    };

    /**
     * DepthMapPublisher is a ROS node which launches OAK-D streams as specified in the config file.
     * Currently it is capable of publishing depth map produced by OAK-D.
     * Streaming AI based objects position is WIP
     */
    class depth_publisher_node
    {
        public:

            std::mutex depth_mutex;
            std::queue<std::pair<float, cv::Mat>> depth_queue;

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

            void clear_depth_queue();

            sensor_msgs::PointCloud2 pcl2ros_converter(pcl::PointCloud<pcl::PointXYZ>::Ptr _pc);

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

        private:
            ros::NodeHandle _nh;

            ros::Timer update_depth, map_timer;
            ros::Publisher depth_image_pub, disparity_image_pub, local_pcl_pub;

            // Class Private Variables
            std::string _mono_resolution;
            std::string _depth_map_topic;

            int _confidence, _lr_check_thres, _median_kernal_size;
            int _subsample; 

            bool _lr_check, _extended, _subpixel;
            bool init = false, registered = false;
        
            double _update_interval;
            double _max_range_clamp, _min_range_clamp;

            double empty_poll_count_time = 0.0;
            double image_width_in_pixels, image_height_in_pixels, focal_length_in_pixels;
            double baseline;
            double hfov, vfov;

            ros::Time node_start_time;

            ros::Publisher _depth_map_pub;

            std::shared_ptr<dai::node::StereoDepth> stereo_depth;
            std::shared_ptr<dai::node::XLinkOut> xout;

            // Define sources and outputs
            std::shared_ptr<dai::node::MonoCamera> mono_left;
            std::shared_ptr<dai::node::MonoCamera> mono_right;

            dai::Pipeline dev_pipeline;
            std::shared_ptr<dai::Device> device;
            std::shared_ptr<dai::DataOutputQueue> disparity_data;

            std::pair<double, cv::Mat> depth_w_stamp;
    };
}