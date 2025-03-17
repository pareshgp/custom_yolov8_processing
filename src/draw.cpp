#include "draw.hpp"
#include "constants.hpp" // for COLORS, CLASS_NAMES
#include <algorithm>
#include <string>
#include <iostream>

// ----------------------------------------------------------------------
// Draw bounding boxes with class labels and scores
// ----------------------------------------------------------------------
void draw_bounding_boxes(cv::Mat &image, const std::vector<Detection> &boxes)
{
    for (const auto &box : boxes)
    {
        int x1 = std::clamp(static_cast<int>(box.x1()), 0, image.cols - 1);
        int y1 = std::clamp(static_cast<int>(box.y1()), 0, image.rows - 1);
        int x2 = std::clamp(static_cast<int>(box.x2()), 0, image.cols - 1);
        int y2 = std::clamp(static_cast<int>(box.y2()), 0, image.rows - 1);

        cv::rectangle(image,
                      cv::Point(x1, y1),
                      cv::Point(x2, y2),
                      COLORS[box.label()],
                      2);

        // Debug print
        /*std::cout << "Detection label index: " 
                  << box.label() << ", Score: " 
                  << box.score() << std::endl;*/

        // Prepare label text
        std::string label = CLASS_NAMES[box.label()] + " " + cv::format("%.2f", box.score());
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, &baseline);

        // Label background
        cv::rectangle(image,
                      cv::Point(x1, y1 - label_size.height - baseline - 5),
                      cv::Point(x1 + label_size.width, y1),
                      COLORS[box.label()],
                      cv::FILLED);

        // Label text
        cv::putText(image, label,
                    cv::Point(x1, y1 - baseline - 5),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ----------------------------------------------------------------------
// Draw instance masks for each detection (overlay color in bounding box)
// ----------------------------------------------------------------------
void draw_masks(cv::Mat &image,
                const std::vector<cv::Mat> &final_masks,
                const std::vector<Detection> &boxes,
                float alpha)
{
    // We expect one mask per detection
    if (final_masks.size() != boxes.size())
    {
        std::cerr << "[draw_masks] Mismatch between number of masks and boxes.\n";
        return;
    }

    for (size_t i = 0; i < boxes.size(); i++)
    {
        int x1 = std::clamp(static_cast<int>(boxes[i].x1()), 0, image.cols - 1);
        int y1 = std::clamp(static_cast<int>(boxes[i].y1()), 0, image.rows - 1);
        int x2 = std::clamp(static_cast<int>(boxes[i].x2()), 0, image.cols - 1);
        int y2 = std::clamp(static_cast<int>(boxes[i].y2()), 0, image.rows - 1);

        int bw = x2 - x1; // bounding box width
        int bh = y2 - y1; // bounding box height
        if (bw <= 0 || bh <= 0)
            continue;

        const cv::Mat &binaryMask = final_masks[i];
        if (binaryMask.empty())
            continue;

        // Ensure we don't exceed image bounds
        cv::Rect bboxRect(x1, y1, bw, bh);
        bboxRect &= cv::Rect(0, 0, image.cols, image.rows);

        // If the mask size doesn't match the bounding box size, warn & skip
        if (binaryMask.rows != bboxRect.height || binaryMask.cols != bboxRect.width)
        {
            std::cerr << "[draw_masks] Mask size does not match box size.\n";
            continue;
        }

        // Choose a color for this class
        cv::Scalar color = COLORS[boxes[i].label()];

        // Blend each pixel if mask == 255
        for (int row = 0; row < bh; ++row)
        {
            const uchar *maskRow = binaryMask.ptr<uchar>(row);
            cv::Vec3b *imgRow    = image.ptr<cv::Vec3b>(y1 + row);

            for (int col = 0; col < bw; ++col)
            {
                if (maskRow[col] == 255)
                {
                    cv::Vec3b &pix = imgRow[x1 + col];
                    // alpha blend in BGR space
                    pix[0] = static_cast<uchar>((1 - alpha) * pix[0] + alpha * color[0]);
                    pix[1] = static_cast<uchar>((1 - alpha) * pix[1] + alpha * color[1]);
                    pix[2] = static_cast<uchar>((1 - alpha) * pix[2] + alpha * color[2]);
                }
            }
        } 
    }
}
