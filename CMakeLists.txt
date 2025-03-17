cmake_minimum_required(VERSION 3.10)
project(yolov8n-seg)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
# Enable AVX support
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
include_directories(${OPENCV_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} include)

file(GLOB SOURCES src/*.cpp)
#add_compile_options(-mavx)

add_executable(yolo_seg ${SOURCES})
target_link_libraries(yolo_seg ${OPENCV_LIBRARIES} ${GST_LIBRARIES} pthread Eigen3::Eigen)



# Place executable outside build folder in `executables/`
set_target_properties(yolo_seg PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executables)
