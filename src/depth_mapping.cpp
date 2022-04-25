#include "ros/ros.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "depth_mapping.h"

namespace depthai_ros
{

    depth_publisher_node::depth_publisher_node(
        ros::NodeHandle &nodeHandle) : _nh(nodeHandle)
    {
        /**
         * @brief https://docs.luxonis.com/projects/api/en/latest/references/cpp/
         * 
         * @param mono_resolution (std::string): Resolution of the camera object
         * @param depth_map_topic_ (std::string): Topic on which depth map will be broadcasted (etc /depth_map)
         * @param confidence (int): Confidence threshold for disparity calculation (Confidence threshold value 0 to 255)
         * @param lr_check (bool): Computes and combines disparities in both L-R and R-L directions, and combine them
         * @param extended (bool): Disparity range increased from 95 to 190, combined from full resolution and downscaled images. Suitable for short range objects 
         * @param subpixel (bool): Better accuracy for longer distance, fractional disparity 32-levels:
         * @param update_interval (double): Interval for timer
         * 
         */
        _nh.param<std::string>("mono_resolution", _mono_resolution, "720p");
        _nh.param<std::string>("depth_map_topic", _depth_map_topic, "/depth");

        _nh.param<int>("confidence", _confidence, 150);
        _nh.param<int>("lr_check_thres", _lr_check_thres, 150);
        _nh.param<int>("median_kernal_size", _median_kernal_size, 7);

        _nh.param<double>("update_interval", _update_interval, 0.1);

        _nh.param<bool>("lr_check", _lr_check, true);
        _nh.param<bool>("extended", _extended, true);
        _nh.param<bool>("subpixel", _subpixel, false);

        depth_image_pub = _nh.advertise<sensor_msgs::Image>(
            _depth_map_topic, 20);

        setup_depth_stream();

        node_start_time = ros::Time::now();

        update_depth = _nh.createTimer(ros::Duration(_update_interval), 
            &depth_publisher_node::update_depth_timer, this, false, false);
        update_depth.start();
    }

    // Destroying 
    depth_publisher_node::~depth_publisher_node() 
    {
        update_depth.stop();
    }

    void depth_publisher_node::update_depth_timer(const ros::TimerEvent &)
    {
        if (!registered)
        {
            if (!device_poll())
            {   
                if (empty_poll_count_time > 10.0)
                {
                    ROS_ERROR("empty_poll_count_time more than 10s");
                    _nh.shutdown();
                    std::exit(EXIT_SUCCESS);
                    return;
                }
                empty_poll_count_time = (ros::Time::now() - node_start_time).toSec();
                return;
            }
        }
        
        if (!init)
        {
            // Try connecting to device and start the pipeline
            device = std::make_shared<dai::Device>(dev_pipeline);
            // Get output queue
            depth_data = device->getOutputQueue("disparity", 4, false);
            init = true;
        }
        
        // Receive 'depth' frame from device
        auto img_depth = depth_data->get<dai::ImgFrame>();
        auto depth_img = img_depth->getFrame();

        // double min_dist, max_dist;
        // minMaxLoc(depth_img, &min_dist, &max_dist); // Find minimum and maximum intensities
        // ROS_INFO("Min %lfmm Max %lfmm", min_dist, max_dist);

        cv_bridge::CvImage bridge_msg;

        bridge_msg.header.stamp = ros::Time::now();
        bridge_msg.header.frame_id = "";

        bridge_msg.image = depth_img;
        bridge_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;

        depth_image_pub.publish(bridge_msg.toImageMsg());
    }

    bool depth_publisher_node::device_poll()
    {
        auto device_info_vec = dai::Device::getAllAvailableDevices();

        for (auto &info : device_info_vec) {
            registered = true;
            std::cout << "Found device with specified id : " << (info.getMxId()).c_str() << std::endl;
            return true;
        }
        ROS_ERROR("Failed to find device in %lfs", (ros::Time::now() - node_start_time).toSec());
        return false;
    }


    void depth_publisher_node::setup_depth_stream()
    {
        mono_left = dev_pipeline.create<dai::node::MonoCamera>();
        mono_right = dev_pipeline.create<dai::node::MonoCamera>();

        dai::node::MonoCamera::Properties::SensorResolution mono_res;

        // 1280 x 720
        if(_mono_resolution == "720p")
        {
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_720_P; 
        }
        // 640 x 400
        else if(_mono_resolution == "400p" )
        {
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_400_P; 
        }
        // 1280 x 800
        else if(_mono_resolution == "800p" )
        {
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_800_P; 
        }
        // 640 x 480
        else if(_mono_resolution == "480p" )
        {
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_480_P; 
        }
        else{
            ROS_ERROR("Invalid parameter : _mono_resolution: %s", _mono_resolution.c_str());
            throw std::runtime_error("Invalid mono camera resolution.");
        }

        mono_left->setResolution(mono_res); mono_right->setResolution(mono_res);
        mono_left->setBoardSocket(dai::CameraBoardSocket::LEFT);
        mono_right->setBoardSocket(dai::CameraBoardSocket::RIGHT);

        stereo_depth = dev_pipeline.create<dai::node::StereoDepth>();
        xout = dev_pipeline.create<dai::node::XLinkOut>();
        
        xout->setStreamName("disparity");

        stereo_depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
        stereo_depth->initialConfig.setConfidenceThreshold(_confidence);
        stereo_depth->initialConfig.setLeftRightCheckThreshold(_lr_check_thres);

        stereo_depth->setLeftRightCheck(_lr_check);
        stereo_depth->setExtendedDisparity(_extended);
        stereo_depth->setSubpixel(_subpixel);

        // MEDIAN_OFF
        if(_median_kernal_size == 0)
        {
            stereo_depth->initialConfig.setMedianFilter(dai::MedianFilter::MEDIAN_OFF);
        }
        // KERNEL_3x3
        else if(_median_kernal_size == 3)
        {
            stereo_depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_3x3);
        }
        // KERNEL_5x5
        else if(_median_kernal_size == 5)
        {
            stereo_depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);
        }
        // KERNEL_7x7
        else if(_median_kernal_size == 7)
        {
            stereo_depth->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_7x7);
        }
        else{
            ROS_ERROR("Invalid parameter : _median_kernal_size: %d", _median_kernal_size);
            throw std::runtime_error("Invalid Median Filter Size");
        }
        // Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)

        // Linking
        mono_left->out.link(stereo_depth->left);
        mono_right->out.link(stereo_depth->right);
        stereo_depth->disparity.link(xout->input);
    }



}