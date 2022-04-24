#include <string>
#include <iostream>
#include <memory>

#include "ros/ros.h"

// include depthai library
#include <depthai/depthai.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>



namespace depthai_ros
{
    using CV_mat_ptr = std::shared_ptr<cv::Mat>;

    /**
     * DepthMapPublisher is a ROS node which launches OAK-D streams as specified in the config file.
     * Currently it is capable of publishing depth map produced by OAK-D.
     * Streaming AI based objects position is WIP
     */
    class depth_publisher_node
    {
        public:
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

            bool device_poll();

        private:
            ros::NodeHandle _nh;

            ros::Timer update_depth;
            ros::Publisher depth_image_pub;

            // Class Private Variables
            std::string _mono_resolution;
            std::string _depth_map_topic;
            int _confidence, _lr_check_thres, _median_kernal_size;
            bool _lr_check, _extended, _subpixel;
            double _update_interval;

            double empty_poll_count_time = 0.0;

            ros::Time node_start_time;

            ros::Publisher _depth_map_pub;

            std::shared_ptr<dai::node::StereoDepth> stereo_depth;
            std::shared_ptr<dai::node::XLinkOut> xout;

            // Define sources and outputs
            std::shared_ptr<dai::node::MonoCamera> mono_left;
            std::shared_ptr<dai::node::MonoCamera> mono_right;

            dai::Pipeline dev_pipeline;
            std::unordered_map<std::string, CV_mat_ptr> _output_streams;
    };

    class depth_mapping_node
    {
        
    };
}