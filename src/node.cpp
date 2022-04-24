#include "depth_mapping.h"

using namespace depthai_ros;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "depthai_mapping_node");

    ros::NodeHandle nh("~");
    // The setpoint publishing rate MUST be faster than 2Hz
    depth_publisher_node depth_publisher_node(nh);
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();
    return 0;
}
