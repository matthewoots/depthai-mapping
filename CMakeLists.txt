cmake_minimum_required(VERSION 3.2)

project(depthai_mapping)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS 
    include

  LIBRARIES 
    ${PROJECT_NAME}

  CATKIN_DEPENDS
    roscpp

  DEPENDS
    OpenCV 
)

set(depthai_DIR "/home/$ENV{USER}/depthai-core/build/install/lib/cmake/depthai")

find_package(depthai CONFIG REQUIRED)
if(depthai_FOUND)
  message(WARNING "depthai FOUND, Version:" ${depthai_VERSION} " directory: "
                  ${depthai_DIR})
endif()

## Build 
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  
)

add_executable(depthai_mapping_node
  src/node.cpp
  src/depth_mapping.cpp
)

target_link_libraries(depthai_mapping_node 
 PUBLIC 
    ${catkin_LIBRARIES}
    depthai::opencv

)

add_dependencies(depthai_mapping_node
  ${catkin_EXPORTED_TARGETS} 
)


