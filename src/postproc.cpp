#include "postproc.hpp"

// Simple helper for elapsed time in ms
double elapsed_ms(
    const std::chrono::high_resolution_clock::time_point &start,
    const std::chrono::high_resolution_clock::time_point &end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ------------------------------------------------------------------------
// PARTIAL MASK GENERATION (Corrected Coordinate Mapping)
/**
 * @brief Generates a partial mask for a given detection.
 *
 * This function maps the bounding box coordinates from the original image to the proto coordinates,
 * computes the dot product of the mask coefficients with the proto data, applies a sigmoid function,
 * resizes the mask to the original bounding box size, and binarizes the mask.
 *
 * @param det The detection object containing bounding box coordinates and mask coefficients.
 * @param protoCHW The proto data in CHW layout (channels, height, width).
 * @param C The number of channels in the proto data.
 * @param H The height of the proto data.
 * @param W The width of the proto data.
 * @param originalWidth The width of the original image.
 * @param originalHeight The height of the original image.
 * @return A binary mask in the original bounding box size.
 */
static cv::Mat generate_partial_mask(
    const Detection &det,
    const std::vector<float> &protoCHW, // CHW layout: [C * H * W]
    int C, int H, int W,                // e.g., 32 × 160 × 160
    float originalWidth, float originalHeight)
{
    // 1) Define scales from original -> proto
    float scale_x_proto = static_cast<float>(W) / originalWidth;
    float scale_y_proto = static_cast<float>(H) / originalHeight;

    // 2) Map bounding box (in original coords) into proto coords
    float x1_proto = det.x1() * scale_x_proto;
    float y1_proto = det.y1() * scale_y_proto;
    float w_proto = det.w() * scale_x_proto;
    float h_proto = det.h() * scale_y_proto;

    // Convert to integer region in [0..W],[0..H]
    int left = std::max(0, static_cast<int>(std::floor(x1_proto)));
    int top = std::max(0, static_cast<int>(std::floor(y1_proto)));
    int right = std::min(W, static_cast<int>(std::ceil(x1_proto + w_proto)));
    int bottom = std::min(H, static_cast<int>(std::ceil(y1_proto + h_proto)));

    if (left >= right || top >= bottom)
    {
        // Invalid region => empty mask
        return cv::Mat();
    }

    int regionW = right - left;
    int regionH = bottom - top;

    // Prepare sub-region in float format
    cv::Mat regionMask(regionH, regionW, CV_32FC1, cv::Scalar(0.f));

    const float *coeffs = det._mask_coeffs.data(); // 32 coeffs
    const int HW = H * W;

    // -----------------------
    // Dot-product in sub-region
    // -----------------------
    for (int ry = 0; ry < regionH; ++ry)
    {
        float *outRow = regionMask.ptr<float>(ry);
        int y_global = top + ry; // row in [0..H)
        for (int rx = 0; rx < regionW; ++rx)
        {
            int x_global = left + rx; // col in [0..W)
            int pixel_offset = y_global * W + x_global;

            float sumVal = 0.f;
#if defined(__AVX__) || defined(__AVX2__)
            // Process 32 channels in 4 chunks of 8
            __m256 sum_vec = _mm256_setzero_ps();
            for (int c = 0; c < C; c += 8)
            {
                const float *pProto = &protoCHW[c * HW + pixel_offset];
                __m256 proto_vec = _mm256_loadu_ps(pProto);
                __m256 coeff_vec = _mm256_loadu_ps(coeffs + c);
                sum_vec = _mm256_fmadd_ps(proto_vec, coeff_vec, sum_vec);
            }
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            sumVal = std::accumulate(sum_array, sum_array + 8, 0.0f);
#else
            // Fallback scalar
            for (int c = 0; c < C; ++c)
            {
                sumVal += protoCHW[c * HW + pixel_offset] * coeffs[c];
            }
#endif
            outRow[rx] = sumVal;
        }
    }

    // -----------------------
    // Sigmoid the region (regionMask = 1 / (1 + exp(-val)))
    // -----------------------
    cv::exp(-regionMask, regionMask);
    regionMask = 1.0f / (1.0f + regionMask);

    // -----------------------
    // Resize sub-region to bounding box size in original coords
    // (det.w() x det.h())
    // -----------------------
    int dstW = static_cast<int>(det.w());
    int dstH = static_cast<int>(det.h());
    if (dstW <= 0 || dstH <= 0)
    {
        return cv::Mat(); // empty
    }

    cv::Mat maskFullSize;
    cv::resize(regionMask, maskFullSize, cv::Size(dstW, dstH), 0, 0, cv::INTER_LINEAR);

    // Binarize
    cv::Mat binaryMask;
    cv::threshold(maskFullSize, binaryMask, 0.5, 255, cv::THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8UC1);

    return binaryMask;
}

// ------------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------------
static int frame_count = 0;
/**
 * @brief Post-processes the YOLO model output to generate detection results.
 *
 * This function takes the raw output tensor from the YOLO model and the original frame,
 * extracts bounding boxes, scores, and mask coefficients, applies non-maximum suppression (NMS),
 * generates partial masks for each detection, and draws the masks and bounding boxes on the original frame.
 *
 * @param input_tensor The raw output tensor from the YOLO model.
 * @param original_frame The original frame from which the detections were made.
 * @return A cv::Mat object containing the original frame with drawn masks and bounding boxes.
 */
cv::Mat yolo_post_process(const std::vector<char> &input_tensor, const cv::Mat &original_frame)
{
    try
    {
        constexpr float scale_x = static_cast<float>(ORIGINAL_WIDTH) / LIMITER_WIDTH;
        constexpr float scale_y = static_cast<float>(ORIGINAL_HEIGHT) / LIMITER_HEIGHT;

        constexpr size_t TENSOR_SIZE_WITHOUT_IMAGE = BBOX_SIZE + SCORES_SIZE + MASK_COEFFS_SIZE + MASKS_SIZE;

        if (input_tensor.size() < TENSOR_SIZE_WITHOUT_IMAGE * sizeof(float))
        {
            throw std::runtime_error("Input tensor size is smaller than expected tensor size.");
        }

        std::vector<float> tensor_data(TENSOR_SIZE_WITHOUT_IMAGE);
        std::memcpy(tensor_data.data(), input_tensor.data(), TENSOR_SIZE_WITHOUT_IMAGE * sizeof(float));
// Parallelize the loop to speed up the copying of mask data into the protoCHW vector
#pragma omp parallel for
        std::span<float> bbox_data(tensor_data.data() + BBOX_OFFSET, BBOX_SIZE);
        std::span<float> scores_data(tensor_data.data() + SCORES_OFFSET, SCORES_SIZE);
        std::span<float> mask_coeffs(tensor_data.data() + MASK_COEFFS_OFFSET, MASK_COEFFS_SIZE);
        std::span<float> masks_data(tensor_data.data() + MASKS_OFFSET, MASKS_SIZE);

        std::vector<float> protoCHW(MASKS_SIZE, 0.f);
        const int HW = H * W;
        const float *masks_ptr = masks_data.data();
        float *output_ptr = protoCHW.data();

#pragma omp parallel for
        for (int c = 0; c < C; ++c)
        {
            float *dst = output_ptr + c * HW;
            const float *src = masks_ptr + c;

#if defined(__AVX2__)
            for (int j = 0; j < HW; j += 8)
            {
                __m256 src_vec = _mm256_set_ps(src[(j + 7) * C], src[(j + 6) * C], src[(j + 5) * C], src[(j + 4) * C], src[(j + 3) * C], src[(j + 2) * C], src[(j + 1) * C], src[j * C]);
                _mm256_storeu_ps(dst + j, src_vec);
            }
#else
            for (int j = 0; j < HW; ++j)
            {
                dst[j] = src[j * C];
            }
#endif
        }

        std::vector<Detection> detections;
        detections.reserve(scores_data.size() / NUM_CLASSES);

        const float *bbox_ptr = bbox_data.data();
        const float *score_ptr = scores_data.data();
        const float *mask_ptr = mask_coeffs.data();

        size_t num_boxes = scores_data.size() / NUM_CLASSES;
        for (size_t j = 0; j < num_boxes; ++j)
        {
            auto max_it = std::max_element(score_ptr, score_ptr + NUM_CLASSES);
            float max_score = *max_it;
            int max_class = static_cast<int>(std::distance(score_ptr, max_it));

            if (max_score >= SCORE_THRESHOLD)
            {
                float x1 = std::clamp(bbox_ptr[0] - bbox_ptr[2] * 0.5f, 0.f, LIMITER_WIDTH) * scale_x;
                float y1 = std::clamp(bbox_ptr[1] - bbox_ptr[3] * 0.5f, 0.f, LIMITER_HEIGHT) * scale_y;
                float w = std::clamp(bbox_ptr[2], 0.f, LIMITER_WIDTH) * scale_x;
                float h = std::clamp(bbox_ptr[3], 0.f, LIMITER_HEIGHT) * scale_y;

                detections.emplace_back(x1, y1, w, h, max_class, max_score);
                std::copy(mask_ptr, mask_ptr + 32, detections.back()._mask_coeffs.begin());
            }
            bbox_ptr += 4;
            score_ptr += NUM_CLASSES;
            mask_ptr += 32;
        }

        std::vector<Detection> final_boxes = apply_nms_topk(detections, NMS_THRESHOLD, TOP_K);
        std::vector<cv::Mat> final_masks(final_boxes.size());

#pragma omp parallel for schedule(dynamic) // Dynamic scheduling is used to balance the workload among threads as the processing time for each mask may vary.

        for (int j = 0; j < static_cast<int>(final_boxes.size()); ++j)
        {
            final_masks[j] = generate_partial_mask(
                final_boxes[j],
                protoCHW,
                C, H, W,
                static_cast<float>(ORIGINAL_WIDTH),
                static_cast<float>(ORIGINAL_HEIGHT));
        }
        cv::Mat rgb_image;
        cv::cvtColor(original_frame, rgb_image, cv::COLOR_YUV2BGR_NV12);
        draw_masks(rgb_image, final_masks, final_boxes, 0.5f);
        draw_bounding_boxes(rgb_image, final_boxes);
        return rgb_image;
    }
    catch (const std::exception &e)
    {
        // Handle any standard exceptions that may occur during post-processing
        std::cerr << "Error: " << e.what() << std::endl;
        return cv::Mat();
    }
}
