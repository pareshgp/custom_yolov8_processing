#include "constants.hpp"

std::vector<std::string> CLASS_NAMES = {"pedestrian", "rider", "car", "truck", "bus","train", "motorcycle", "bicycle", "trailer", "caravan"};

// std::vector<cv::Scalar> COLORS = {
//     cv::Scalar(56, 62, 150), cv::Scalar(180, 130, 70), cv::Scalar(120, 120, 120),
//     cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
//     cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
//     cv::Scalar(100, 100, 200)};

std::vector<cv::Scalar> COLORS = {
    cv::Scalar(220, 20, 60),   // Crimson
    cv::Scalar(0, 191, 255),   // Deep Sky Blue
    cv::Scalar(34, 139, 34),   // Forest Green
    cv::Scalar(255, 140, 0),   // Dark Orange
    cv::Scalar(128, 0, 128),   // Purple
    cv::Scalar(255, 215, 0),   // Gold
    cv::Scalar(75, 0, 130),    // Indigo
    cv::Scalar(255, 20, 147),  // Deep Pink
    cv::Scalar(0, 255, 127),   // Spring Green
    cv::Scalar(70, 130, 180)   // Steel Blue
};