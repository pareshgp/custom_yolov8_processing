#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Constants
constexpr float LIMITER_WIDTH = 640.0f;
constexpr float LIMITER_HEIGHT = 640.0f;
constexpr float ORIGINAL_WIDTH = 1920.0f;
constexpr float ORIGINAL_HEIGHT = 1080.0f;

constexpr float SCORE_THRESHOLD = 0.15f;
constexpr float NMS_THRESHOLD = 0.5f;
constexpr int TOP_K = 25;
constexpr int NUM_CLASSES = 10;

// Offsets/sizes for raw tensor data (example placeholders)
constexpr uint32_t BBOX_OFFSET = 0;
constexpr uint32_t BBOX_SIZE = 33600;
constexpr uint32_t SCORES_OFFSET = 33600;
constexpr uint32_t SCORES_SIZE = 84000;
constexpr uint32_t MASK_COEFFS_OFFSET = 117600;
constexpr uint32_t MASK_COEFFS_SIZE = 268800;
constexpr uint32_t MASKS_OFFSET = 386400;
constexpr uint32_t MASKS_SIZE = 819200;

// Height, Width
constexpr int H = 160;
constexpr int W = 160;
constexpr int C = 32;

// Global class names (declared "extern" here, defined in constants.cpp)
extern std::vector<std::string> CLASS_NAMES;

// Global colors for each class
extern std::vector<cv::Scalar> COLORS;

#endif // CONSTANTS_HPP
