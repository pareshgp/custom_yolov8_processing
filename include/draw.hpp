#ifndef DRAW_HPP
#define DRAW_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "detection.hpp"

// Draw bounding boxes + label
void draw_bounding_boxes(cv::Mat &image, const std::vector<Detection> &boxes);

// Draw final masks (semi-transparent overlay)
void draw_masks(cv::Mat &image,
                const std::vector<cv::Mat> &final_masks,
                const std::vector<Detection> &boxes,
                float alpha = 0.5f);

#endif // DRAW_HPP
