#include "ros/ros.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc/disparity_filter.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>


#include "depth_mapping.h"

namespace depthai_ros
{

    /**
     * @brief For depth_publisher_node class
     */

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
        _nh.param<int>("mean_k", _mean_k, 50);
        _nh.param<int>("subsample", _subsample, 1);

        _nh.param<double>("max_range_clamp", _max_range_clamp, 5.0);
        _nh.param<double>("update_interval", _update_interval, 0.1);
        _nh.param<double>("downsample", _downsample, 1.0);
        _nh.param<double>("std_dev_mul", _std_dev_mul, 1.0);
        _nh.param<double>("lambda", _lambda, 1.0);
        _nh.param<double>("sigma", _sigma, 1.0);
        
        _nh.param<bool>("lr_check", _lr_check, true);
        _nh.param<bool>("extended", _extended, true);
        _nh.param<bool>("subpixel", _subpixel, false);

        

        depth_image_pub = _nh.advertise<sensor_msgs::Image>(
            _depth_map_topic, 3);
        disparity_image_pub = _nh.advertise<sensor_msgs::Image>(
            "/disparity", 3);
        local_pcl_pub = _nh.advertise<sensor_msgs::PointCloud2>(
            "/pcl", 3);

        setup_depth_stream();

        node_start_time = ros::Time::now();

        update_depth = _nh.createTimer(ros::Duration(_update_interval), 
            &depth_publisher_node::update_depth_timer, this, false, false);

        map_timer = _nh.createTimer(ros::Duration(_update_interval), 
            &depth_publisher_node::update_map, this, false, false);

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
            disparity_data = device->getOutputQueue("disparity_stream", 4, false);
            right_data = device->getOutputQueue("right_mono_stream", 4, false);
            
            // Start with init mapping node
            map_node.init(_min_range_clamp, _max_range_clamp, 
                _subsample, _mean_k, _std_dev_mul);
            map_node.intrinsics_ = intrinsics;

            map_timer.start();
            
            init = true;
        }
        
        // Receive 'disparity' frame from device
        auto img_disparity = disparity_data->get<dai::ImgFrame>();
        // Receive 'right' frame from device
        auto img_right = right_data->get<dai::ImgFrame>();
        cv::Mat right_mat = img_right->getFrame();

        // Now mat is in 0 to 190 (extended) or 0 to 95
        // Format is 16UC1
        cv::Mat mat = img_disparity->getFrame();

        cv_bridge::CvImage disparity_msg;
        disparity_msg.header.stamp = ros::Time::now();
        disparity_msg.header.frame_id = "";

        disparity_msg.image = mat;

        // WLS Filtering
        cv::Mat filtered_disp_mat;
        double lambda = _lambda * 100.0;
        double sigma = _sigma / 10.0;
        filtered_disp_mat = disparity_filter(mat, right_mat, lambda, sigma);

        disparity_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
        int rows = mat.rows;
        int cols = mat.cols;
        ROS_INFO("height %dpix width %dpix", 
            rows, cols);

        disparity_image_pub.publish(disparity_msg.toImageMsg());

        filtered_disp_mat.convertTo(filtered_disp_mat, CV_32FC1);        

        // https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/
        // depth = focal_length_in_pixels * baseline / disparity_in_pixels
        // focal_length_in_pixels = image_width_in_pixels * 0.5 / tan(HFOV * 0.5 * PI/180)        

        // Depth is in cm
        auto depth_img_ori = (focal_length_in_pixels / _downsample) * baseline / filtered_disp_mat;
        // Depth in m
        depth_img_ori = depth_img_ori / 100.0;
        depth_img_ori = min(depth_img_ori, _max_range_clamp);

        cv::Mat depth_img;

        resize(depth_img_ori, depth_img, cv::Size(
            image_width_in_pixels, image_height_in_pixels),
            cv::INTER_NEAREST);

        update_depth_pair(depth_img);

        // Add to the queue
        // odom_queue.push(odom_tf_w_stamp);

        // double min_dist, max_dist;
        // minMaxLoc(depth_img, &min_dist, &max_dist); // Find minimum and maximum intensities
        // ROS_INFO("Min %lfm Max %lfm\n", min_dist, max_dist);

        cv_bridge::CvImage bridge_msg;

        bridge_msg.header.stamp = ros::Time::now();
        bridge_msg.header.frame_id = "";

        bridge_msg.image = depth_img;
        bridge_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;

        depth_image_pub.publish(bridge_msg.toImageMsg());
    }

    void depth_publisher_node::update_map(const ros::TimerEvent &)
    {
        if (!check_depth_mat())
            return;

        map_node.calc_pcl_pointcloud(get_depth_pair());

        sensor_msgs::PointCloud2 local_pcl_msg;

        pcl::toROSMsg(*map_node.local_pc, local_pcl_msg);

        local_pcl_msg.header.stamp = ros::Time::now();
        local_pcl_msg.header.frame_id = "/base_link";
        local_pcl_pub.publish(local_pcl_msg); 
    }

    bool depth_publisher_node::device_poll()
    {
        auto device_info_vec = dai::Device::getAllAvailableDevices();

        for (auto &info : device_info_vec) {
            registered = true;
            std::cout << "Found device" << std::endl;
            std::cout << "with specified id : " << (info.getMxId()).c_str() << std::endl;
            
            return true;
        }
        ROS_ERROR("Failed to find device in %lfs", (ros::Time::now() - node_start_time).toSec());
        return false;
    }



    cv::Mat depth_publisher_node::disparity_filter(
        cv::Mat disparity, cv::Mat right, double lambda, double sigma)
    {
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
        
        wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        cv::Mat filtered;
        // CV_WRAP virtual void filter(InputArray disparity_map_left,
        // InputArray left_view, OutputArray filtered_disparity_map
        wls_filter->filter(disparity, right, filtered);

        return filtered;
    }


    void depth_publisher_node::setup_depth_stream()
    {
        int decimation = 4;
        double decimation_factor = 1.0 / (double)decimation;

        _downsample = decimation_factor;

        mono_left = dev_pipeline.create<dai::node::MonoCamera>();
        mono_right = dev_pipeline.create<dai::node::MonoCamera>();

        dai::node::MonoCamera::Properties::SensorResolution mono_res;
        baseline = 7.5;
        hfov = 72.0; vfov = 50.0;

        // 1280 x 720
        if(_mono_resolution == "720p")
        {
            image_width_in_pixels = 1280.0 * _downsample;
            image_height_in_pixels = 720.0 * _downsample;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_720_P; 
        }
        // 640 x 400
        else if(_mono_resolution == "400p" )
        {
            image_width_in_pixels = 640.0 * _downsample;
            image_height_in_pixels = 400.0 * _downsample;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_400_P; 
        }
        // 1280 x 800
        else if(_mono_resolution == "800p" )
        {
            image_width_in_pixels = 1280.0 * _downsample;
            image_height_in_pixels = 800.0 * _downsample;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_800_P; 
        }
        // 640 x 480
        else if(_mono_resolution == "480p" )
        {
            image_width_in_pixels = 640.0 * _downsample;
            image_height_in_pixels = 480.0 * _downsample;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_480_P; 
        }
        else{
            ROS_ERROR("Invalid parameter : _mono_resolution: %s", _mono_resolution.c_str());
            throw std::runtime_error("Invalid mono camera resolution.");
        }

        // With xxxxP mono camera resolution where HFOV = 71.9 degrees
        focal_length_in_pixels = image_width_in_pixels * 0.5 / 
            tan(hfov * 0.5 * M_PI / 180.0);        

        mono_left->setResolution(mono_res); mono_right->setResolution(mono_res);
        mono_left->setBoardSocket(dai::CameraBoardSocket::LEFT);
        mono_right->setBoardSocket(dai::CameraBoardSocket::RIGHT);

        stereo_depth = dev_pipeline.create<dai::node::StereoDepth>();
        xout_disp = dev_pipeline.create<dai::node::XLinkOut>();
        xout_right = dev_pipeline.create<dai::node::XLinkOut>();
        
        xout_disp->setStreamName("disparity_stream");
        xout_right->setStreamName("right_mono_stream");

        stereo_depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_ACCURACY);
        stereo_depth->initialConfig.setConfidenceThreshold(_confidence);
        stereo_depth->initialConfig.setLeftRightCheckThreshold(_lr_check_thres);

        stereo_depth->setLeftRightCheck(_lr_check);
        stereo_depth->setExtendedDisparity(_extended);
        stereo_depth->setSubpixel(_subpixel);

        auto PostConfig = stereo_depth->initialConfig.get();

        // Explanation : https://dev.intelrealsense.com/docs/depth-post-processing
        // Deafult values : https://dev.intelrealsense.com/docs/post-processing-filters
        PostConfig.postProcessing.speckleFilter.enable = true;

        PostConfig.postProcessing.speckleFilter.speckleRange = 300;

        PostConfig.postProcessing.temporalFilter.enable = true;
        // This is default
        // PostConfig.postProcessing.temporalFilter.persistencyMode = 
        //     dai::RawStereoDepthConfig::PostProcessing::TemporalFilter::PersistencyMode::VALID_2_IN_LAST_4;
        
        // Following realsense D435i
        PostConfig.postProcessing.temporalFilter.alpha = 0.5;
        PostConfig.postProcessing.temporalFilter.delta = 20;

        PostConfig.postProcessing.spatialFilter.enable = true;
        // Following realsense D435i
        PostConfig.postProcessing.spatialFilter.holeFillingRadius = 4;
        PostConfig.postProcessing.spatialFilter.numIterations = 2;
        PostConfig.postProcessing.spatialFilter.alpha = 0.5;
        PostConfig.postProcessing.spatialFilter.delta = 20;

        PostConfig.postProcessing.thresholdFilter.minRange = 400;
        PostConfig.postProcessing.thresholdFilter.maxRange = 15000;

        // Following realsense D435i
        PostConfig.postProcessing.decimationFilter.decimationFactor = decimation;
        PostConfig.postProcessing.decimationFilter.decimationMode = 
            dai::RawStereoDepthConfig::PostProcessing::DecimationFilter::DecimationMode::NON_ZERO_MEDIAN;

        stereo_depth->initialConfig.set(PostConfig);

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
        stereo_depth->disparity.link(xout_disp->input);
        stereo_depth->syncedRight.link(xout_right->input);

        _min_range_clamp = focal_length_in_pixels * baseline / stereo_depth->initialConfig.getMaxDisparity();
        _min_range_clamp = _min_range_clamp/100.0;

        auto focal_length_v_in_pixels = image_height_in_pixels * 0.5 / 
            tan(vfov * 0.5 * M_PI / 180.0) ; 

        ROS_INFO("Horizontal %lfpix Vertical %lfpix resolution", 
            image_width_in_pixels / _downsample, image_height_in_pixels / _downsample);
        ROS_INFO("[After Downsample] Horizontal %lfpix Vertical %lfpix resolution", 
            image_width_in_pixels, image_height_in_pixels);
        ROS_INFO("Max disparity %lfpix", stereo_depth->initialConfig.getMaxDisparity());
        ROS_INFO("[After Downsample] Focal length horizontal %lfpix vertical %lfpix", 
            focal_length_in_pixels, focal_length_v_in_pixels);
        ROS_INFO("Baseline %lfcm, HFOV %lfdeg", baseline, hfov);
        ROS_INFO("Min stereo distance %lfm", _min_range_clamp);

        // Set up intrinsics
        /**
         * camera intrinsics (K)
            * |f_u   0   o_u|
            * | 0   f_v  0_v|
            * | 0    0    1 |
         * 
         */

        intrinsics << focal_length_in_pixels, 0, image_width_in_pixels/2,
                    0, focal_length_v_in_pixels, image_height_in_pixels/2,
                    0, 0, 1;
    }

}