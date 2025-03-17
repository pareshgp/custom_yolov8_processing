#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>

// Forward declaration for usage of cv::Mat in function signatures
namespace cv
{
    class Mat;
}

struct IntermediateBox
{
    float _cx;
    float _cy;
    float _w;
    float _h;
};

struct Detection
{
    unsigned int _x1;
    unsigned int _y1;
    unsigned int _w;
    unsigned int _h;
    unsigned int _label;
    float _score;
    std::array<float, 32> _mask_coeffs; // 32 mask coefficients

    constexpr float score() const noexcept { return _score; }
    constexpr unsigned int x1() const noexcept { return _x1; }
    constexpr unsigned int y1() const noexcept { return _y1; }
    constexpr unsigned int w() const noexcept { return _w; }
    constexpr unsigned int h() const noexcept { return _h; }
    constexpr unsigned int x2() const noexcept { return _x1 + _w; }
    constexpr unsigned int y2() const noexcept { return _y1 + _h; }
    constexpr unsigned int label() const noexcept { return _label; }
    constexpr const std::array<float, 32> &mask_coeff() const noexcept { return _mask_coeffs; }
};

// IoU function
float IoU(const Detection &a, const Detection &b);

// Non-maximum suppression with Top-K filtering
std::vector<Detection> apply_nms_topk(std::vector<Detection> &boxes, float iou_threshold, size_t k);

#endif // DETECTION_HPP
