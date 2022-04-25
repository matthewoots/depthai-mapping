#include "ros/ros.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

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

        _nh.param<double>("max_range_clamp", _max_range_clamp, 5.0);
        _nh.param<double>("update_interval", _update_interval, 0.1);

        _nh.param<double>("downsample", _downsample, 1.0);

        _nh.param<bool>("lr_check", _lr_check, true);
        _nh.param<bool>("extended", _extended, true);
        _nh.param<bool>("subpixel", _subpixel, false);

        _nh.param<int>("subsample", _subsample, 1);

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
            disparity_data = device->getOutputQueue("disparity", 4, false);
            
            // Start with init mapping node
            map_node.init(_min_range_clamp, _max_range_clamp, _subsample);
            map_node.intrinsics_ = intrinsics;

            map_timer.start();
            
            init = true;
        }
        
        // Receive 'depth' frame from device
        auto img_disparity = disparity_data->get<dai::ImgFrame>();
        // Now mat is in 0 to 190 (extended) or 0 to 95
        // Format is 16UC1
        cv::Mat mat = img_disparity->getFrame();

        cv_bridge::CvImage disparity_msg;
        disparity_msg.header.stamp = ros::Time::now();
        disparity_msg.header.frame_id = "";

        disparity_msg.image = mat;
        disparity_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
        disparity_image_pub.publish(disparity_msg.toImageMsg());

        mat.convertTo(mat, CV_32FC1);        

        // https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/
        // depth = focal_length_in_pixels * baseline / disparity_in_pixels
        // focal_length_in_pixels = image_width_in_pixels * 0.5 / tan(HFOV * 0.5 * PI/180)        

        // Depth is in cm
        auto depth_img_ori = focal_length_in_pixels * baseline / mat;
        // Depth in m
        depth_img_ori = depth_img_ori / 100.0;
        depth_img_ori = min(depth_img_ori, _max_range_clamp);

        cv::Mat depth_img;

        resize(depth_img_ori, depth_img, cv::Size(
            image_width_in_pixels * _downsample, image_height_in_pixels * _downsample),
            cv::INTER_NEAREST);

        update_depth_pair(depth_img);

        // Add to the queue
        // depth_queue.push(depth_w_stamp);

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
        local_pcl_msg = pcl2ros_converter(map_node.local_pc);

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

    void depth_publisher_node::clear_depth_queue()
    {
        std::lock_guard<std::mutex> d_lock(depth_mutex);
        while (!depth_queue.empty()) 
            depth_queue.pop();
    }

    /** 
    * @brief Convert point cloud from ROS sensor message to 
    * pcl point ptr
    */
    sensor_msgs::PointCloud2
        depth_publisher_node::pcl2ros_converter(pcl::PointCloud<pcl::PointXYZ>::Ptr _pc)
    {
        sensor_msgs::PointCloud2 ros_msg;
        pcl::toROSMsg(*_pc, ros_msg);
        return ros_msg;
    }


    void depth_publisher_node::setup_depth_stream()
    {
        mono_left = dev_pipeline.create<dai::node::MonoCamera>();
        mono_right = dev_pipeline.create<dai::node::MonoCamera>();

        dai::node::MonoCamera::Properties::SensorResolution mono_res;
        baseline = 7.5;
        hfov = 72.0; vfov = 50.0;

        // 1280 x 720
        if(_mono_resolution == "720p")
        {
            image_width_in_pixels = 1280.0;
            image_height_in_pixels = 720.0;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_720_P; 
        }
        // 640 x 400
        else if(_mono_resolution == "400p" )
        {
            image_width_in_pixels = 640.0;
            image_height_in_pixels = 400.0;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_400_P; 
        }
        // 1280 x 800
        else if(_mono_resolution == "800p" )
        {
            image_width_in_pixels = 1280.0;
            image_height_in_pixels = 800.0;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_800_P; 
        }
        // 640 x 480
        else if(_mono_resolution == "480p" )
        {
            image_width_in_pixels = 640.0;
            image_height_in_pixels = 480.0;
            mono_res = dai::node::MonoCamera::Properties::SensorResolution::THE_480_P; 
        }
        else{
            ROS_ERROR("Invalid parameter : _mono_resolution: %s", _mono_resolution.c_str());
            throw std::runtime_error("Invalid mono camera resolution.");
        }

        // With xxxxP mono camera resolution where HFOV = 71.9 degrees
        focal_length_in_pixels = image_width_in_pixels * _downsample * 0.5 / 
            tan(hfov * 0.5 * M_PI / 180.0);        

        mono_left->setResolution(mono_res); mono_right->setResolution(mono_res);
        mono_left->setBoardSocket(dai::CameraBoardSocket::LEFT);
        mono_right->setBoardSocket(dai::CameraBoardSocket::RIGHT);

        stereo_depth = dev_pipeline.create<dai::node::StereoDepth>();
        xout = dev_pipeline.create<dai::node::XLinkOut>();
        
        xout->setStreamName("disparity");

        stereo_depth->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_ACCURACY);
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

        _min_range_clamp = focal_length_in_pixels * baseline / stereo_depth->initialConfig.getMaxDisparity();

        ROS_INFO("Horizontal resolution %lfpix", image_width_in_pixels);
        ROS_INFO("Max disparity %lfpix", stereo_depth->initialConfig.getMaxDisparity());
        ROS_INFO("Focal length horizontal %lfpix", focal_length_in_pixels);
        ROS_INFO("Baseline %lfcm, HFOV %lfdeg", baseline, hfov);
        ROS_INFO("Min stereo distance %lfm", _min_range_clamp/100.0);

        // Set up intrinsics
        /**
         * camera intrinsics (K)
            * |f_u   0   o_u|
            * | 0   f_v  0_v|
            * | 0    0    1 |
         * 
         */
        auto focal_length_v_in_pixels = image_height_in_pixels * _downsample * 0.5 / 
            tan(vfov * 0.5 * M_PI / 180.0) ; 

        intrinsics << focal_length_in_pixels, 0, image_width_in_pixels/2 * _downsample,
                    0, focal_length_v_in_pixels, image_height_in_pixels/2 * _downsample,
                    0, 0, 1;
    }

    


    /**
     * @brief For depth_mapping_node class
     */

    void depth_mapping_node::init(double ll, double ul, double sub_fact)
    {
        depth_lower_limit = ll;
        depth_upper_limit = ul;
        subsample_factor = sub_fact;
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
            if (points_3d(2, i) < (float)depth_upper_limit)
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

        int mean_k = 4; 
        float std_dev_mul = 4.0;
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