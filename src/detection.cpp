#include "detection.hpp"

float IoU(const Detection &a, const Detection &b)
{
    float x1 = std::max(static_cast<float>(a.x1()), static_cast<float>(b.x1()));
    float y1 = std::max(static_cast<float>(a.y1()), static_cast<float>(b.y1()));
    float x2 = std::min(static_cast<float>(a.x2()), static_cast<float>(b.x2()));
    float y2 = std::min(static_cast<float>(a.y2()), static_cast<float>(b.y2()));

    float inter_w = std::max(0.f, x2 - x1);
    float inter_h = std::max(0.f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float a_area = static_cast<float>(a.w()) * static_cast<float>(a.h());
    float b_area = static_cast<float>(b.w()) * static_cast<float>(b.h());

    float union_area = a_area + b_area - inter_area;
    return (union_area > 0.f) ? (inter_area / union_area) : 0.0f;
}

std::vector<Detection> apply_nms_topk(std::vector<Detection> &boxes, float iou_threshold, size_t k)
{
    std::sort(boxes.begin(), boxes.end(),
              [](const Detection &a, const Detection &b)
              {
                  return a.label() < b.label();
              });

    std::vector<Detection> result;
    result.reserve(std::min(k, boxes.size()));

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size() && result.size() < k; ++i)
    {
        if (suppressed[i])
            continue;

        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (!suppressed[j] && IoU(boxes[i], boxes[j]) > iou_threshold)
            {
                suppressed[j] = true;
            }
        }
    }
    return result;
}