#ifndef POSTPROC_HPP
#define POSTPROC_HPP

#include <iostream>
#include <chrono>
#include <fstream>
#include <span>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>  
#include <omp.h>

// OpenCV
#include <opencv2/opencv.hpp>

// For AVX intrinsics if available
#include <immintrin.h>


#include "constants.hpp"    // e.g. BBOX_SIZE, SCORES_SIZE, etc.
#include "utils.hpp"        // e.g. read_tensor_data() or other
#include "detection.hpp"    // struct Detection { float _mask_coeffs[32], etc. }
#include "draw.hpp"

cv::Mat yolo_post_process(const std::vector<char>& input_tensor, const cv::Mat& original_frame);

#endif // POSTPROC_HPP