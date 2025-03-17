#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <fstream>
#include <unordered_map>
#include "postproc.hpp"

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define UDP_FRAMERATE 16
#define VIDEO_READ_INTERVAL 38
#define UDP_START_PORT 8000
#define X264_BITRATE 7000

// Pushing Frames from videothreads to gstreamer pipeline
std::queue<std::tuple<cv::Mat, guint64, int>> frameQueue;
std::mutex frameMutex;
std::condition_variable frameCond;
// Storing Frames
std::unordered_map<int, std::unordered_map<guint, cv::Mat>> frame_storage;

cv::Mat convertYUYVtoNV12(const cv::Mat &yuyv_frame)
{
    int width = yuyv_frame.cols;
    int height = yuyv_frame.rows;

    // Allocate space for NV12 buffer
    cv::Mat nv12(height + height / 2, width, CV_8UC1);

    // Pointers for Y and UV planes
    uint8_t *y_plane = nv12.data;
    uint8_t *uv_plane = nv12.data + (width * height);

    const uint8_t *yuyv = yuyv_frame.data;

    // Convert YUYV to NV12
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j += 2)
        {
            // YUYV -> Y0 U Y1 V
            uint8_t y0 = yuyv[i * width * 2 + j * 2];
            uint8_t u = yuyv[i * width * 2 + j * 2 + 1];
            uint8_t y1 = yuyv[i * width * 2 + j * 2 + 2];
            uint8_t v = yuyv[i * width * 2 + j * 2 + 3];

            // Write Y plane
            y_plane[i * width + j] = y0;
            y_plane[i * width + j + 1] = y1;

            // Write UV plane (interleaved)
            if (i % 2 == 0)
            {
                uv_plane[(i / 2) * width + j] = u;
                uv_plane[(i / 2) * width + j + 1] = v;
            }
        }
    }

    return nv12;
}

cv::Mat convertBGRToNV12(const cv::Mat &rgb_image)
{
    cv::Mat yuv420, y_plane, uv_plane, nv12_frame;
    int width = rgb_image.cols;
    int height = rgb_image.rows;

    // Convert BGR to YUV420 (I420)
    cvtColor(rgb_image, yuv420, cv::COLOR_BGR2YUV_I420);

    // Extract Y plane (Full resolution)
    y_plane = yuv420(cv::Rect(0, 0, width, height)).clone();

    // Compute U and V plane positions
    uchar *u_ptr = yuv420.data + (width * height); // U starts after Y
    uchar *v_ptr = u_ptr + ((width * height) / 4); // V starts after U

    // Allocate UV plane for NV12 (height/2, width)
    uv_plane = cv::Mat(height / 2, width, CV_8UC1);

    // Interleave U and V to create NV12 UV plane
    for (int i = 0; i < height / 2; i++)
    {
        for (int j = 0; j < width / 2; j++)
        {
            uv_plane.at<uchar>(i, j * 2) = u_ptr[i * (width / 2) + j];     // U
            uv_plane.at<uchar>(i, j * 2 + 1) = v_ptr[i * (width / 2) + j]; // V
        }
    }

    // Ensure both planes have the same number of columns
    if (y_plane.cols == uv_plane.cols && y_plane.type() == uv_plane.type())
    {
        vconcat(y_plane, uv_plane, nv12_frame);
    }
    else
    {
        std::cerr << "Error: Mismatched matrix sizes, skipping frame!" << std::endl;
    }

    return nv12_frame;
}

class VideoReaderUSB
{
public:
    VideoReaderUSB(const std::string &source)
    {
        std::string pipeline = "v4l2src device=" + source +
                               " ! video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1 "
                               " ! appsink";

        cap.open(pipeline, cv::CAP_GSTREAMER);
        // cap.open(source, cv::CAP_V4L2);

        if (!cap.isOpened())
        {
            throw std::runtime_error("Failed to open video source");
        }
    }

    bool read(cv::Mat &frame)
    {
        return cap.read(frame);
    }

private:
    cv::VideoCapture cap;
};

class VideoReader
{
public:
    VideoReader(const std::string &source)
    {
        cap.open(source);
        if (!cap.isOpened())
        {
            throw std::runtime_error("Failed to open video source");
        }
    }

    bool read(cv::Mat &frame)
    {
        return cap.read(frame);
    }

private:
    cv::VideoCapture cap;
};

class VideoStreamer
{
public:
    int context_id;
    VideoStreamer(int ctx)
    {
        gst_init(nullptr, nullptr);
        context_id = ctx;
        pipeline = gst_pipeline_new("streaming-pipeline");
        appsrc = gst_element_factory_make("appsrc", "video-source");
        queue1 = gst_element_factory_make("queue", "queue1");
        queue2 = gst_element_factory_make("queue", "queue2");
        queue3 = gst_element_factory_make("queue", "queue3");
        x264enc = gst_element_factory_make("x264enc", "x264enc");
        h264parse = gst_element_factory_make("h264parse", "h264parse");
        rtph264pay = gst_element_factory_make("rtph264pay", "rtph264pay"); // New RTP payload
        udpsink = gst_element_factory_make("udpsink", "udpsink");          // UDP sink

        if (!pipeline || !appsrc || !queue1 || !queue2 || !queue3 ||
            !x264enc || !h264parse || !rtph264pay || !udpsink)
        {
            throw std::runtime_error("Failed to create GStreamer elements");
        }

        // Set properties for elements
        GstCaps *appsrc_caps = gst_caps_new_simple("video/x-raw",
                                                   "format", G_TYPE_STRING, "NV12",
                                                   "width", G_TYPE_INT, IMAGE_WIDTH,
                                                   "height", G_TYPE_INT, IMAGE_HEIGHT,
                                                   "framerate", GST_TYPE_FRACTION, UDP_FRAMERATE, 1,
                                                   NULL);
        g_object_set(appsrc, "caps", appsrc_caps, "format", GST_FORMAT_TIME, "is-live", TRUE, NULL);
        gst_caps_unref(appsrc_caps);

        g_object_set(x264enc, "bitrate", X264_BITRATE, "speed-preset", 1, "tune", 4, NULL);
        int port = UDP_START_PORT + context_id;
        g_object_set(udpsink, "host", "127.0.0.1", "port", port, "sync", FALSE, "async", FALSE, NULL);

        // Add elements to the pipeline
        gst_bin_add_many(GST_BIN(pipeline), appsrc, queue1, x264enc, queue2, h264parse, queue3, rtph264pay, udpsink, NULL);

        // Link elements
        if (!gst_element_link_many(appsrc, queue1, x264enc, queue2, h264parse, queue3, rtph264pay, udpsink, NULL))
        {
            throw std::runtime_error("Failed to link GStreamer elements");
        }

        // Start the pipeline
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
    }

    void pushFrame(const cv::Mat &frame)
    {
        if (frame.empty())
            return;

        GstBuffer *buffer = gst_buffer_new_allocate(NULL, frame.total() * frame.elemSize(), NULL);
        GstMapInfo map;
        GstFlowReturn ret;
        if (gst_buffer_map(buffer, &map, GST_MAP_WRITE))
        {
            std::memcpy(map.data, frame.data, frame.total() * frame.elemSize());
            gst_buffer_unmap(buffer, &map);
        }
        else
        {
            std::cerr << "Error mapping buffer" << std::endl;
            gst_buffer_unref(buffer);
            return;
        }
        if (appsrc)
        {
            g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
            if (ret != GST_FLOW_OK)
            {
                std::cerr << "Error pushing buffer to appsrc: " << gst_flow_get_name(ret) << std::endl;
                gst_buffer_unref(buffer);
            }
        }
        else
        {
            std::cerr << "Error: appsrc is not initialized" << std::endl;
            gst_buffer_unref(buffer);
        }
        gst_buffer_unref(buffer);
    }

    ~VideoStreamer()
    {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }

private:
    GstElement *pipeline, *appsrc, *queue1, *queue2, *queue3, *x264enc, *rtph264pay, *udpsink, *h264parse;
};

std::vector<std::unique_ptr<VideoStreamer>> udpstreamers;

class GStreamerPipeline
{
public:
    GStreamerPipeline()
    {
        gst_init(nullptr, nullptr);
        pipeline = gst_pipeline_new("yolov8-pipeline");
        appsrc = gst_element_factory_make("appsrc", "appsrc");
        appsink = gst_element_factory_make("appsink", "appsink");
        pciehost = gst_element_factory_make("pciehost", "pciehost");
        g_object_set(pciehost, "showfps", 1, "buffersize", 4822400, NULL);

        g_object_set(appsrc, "format", GST_FORMAT_TIME, "is-live", TRUE, NULL);
        g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, NULL);
        g_signal_connect(GST_APP_SINK(appsink), "new-sample", G_CALLBACK(onNewSample), this);

        GstElement *queue1 = gst_element_factory_make("queue", "queue1");
        GstElement *queue2 = gst_element_factory_make("queue", "queue2");
        if (!queue1 || !queue2)
        {
            throw std::runtime_error("Failed to create GStreamer queue elements");
        }

        gst_bin_add_many(GST_BIN(pipeline), appsrc, queue1, pciehost, queue2, appsink, NULL);
        gst_element_link_many(appsrc, queue1, pciehost, queue2, appsink, NULL);
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
    }

    // Attach metadata to a buffer
    void gst_buffer_add_frame_id_meta(GstBuffer *buffer, guint64 frame_id, int context_id)
    {
        // g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);
        meta = gst_buffer_add_custom_meta(buffer, "GstCustomHostMeta");
        if (meta == NULL)
        {
            GST_ERROR("Unable to add metadata info to the buffer");
        }
        else
        {
            meta_struct = gst_custom_meta_get_structure(meta);
            if (meta_struct != NULL)
            {
                gst_structure_set(meta_struct, "frame-id",
                                  G_TYPE_UINT, frame_id, NULL);
                gst_structure_set(meta_struct, "stream-id",
                                  G_TYPE_UINT, context_id, NULL);
            }
            else
            {
                GST_ERROR("Unable to add metadata to the buffer");
            }
        }
    }

    // Retrieve metadata from a buffer
    std::pair<guint, int> gst_buffer_get_frame_id_meta(GstBuffer *buffer)
    {
        guint frame_id = -1;
        guint context_id = -1;
        meta = gst_buffer_get_custom_meta(buffer, "GstCustomHostMeta");
        if (meta != NULL)
        {
            meta_struct = gst_custom_meta_get_structure(meta);
            if (meta_struct == NULL)
            {
                GST_INFO("No metadata attached in buffer");
            }
            else
            {
                gst_structure_get_uint(meta_struct, "frame-id", &frame_id);
                gst_structure_get_uint(meta_struct, "stream-id", &context_id);
                GST_INFO("Frame ID: %d, Stream ID: %d", frame_id, context_id);
            }
        }

        return {frame_id, context_id};
    }

    void processFrames()
    {
        while (running)
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCond.wait(lock, [this]
                           { return !frameQueue.empty() || !running; });

            if (!running)
                break;

            // Retrieve frame and ID from queue
            auto frame_data = std::move(frameQueue.front());
            frameQueue.pop();
            lock.unlock();

            cv::Mat frame = std::get<0>(frame_data);
            guint64 frame_id = std::get<1>(frame_data);
            int context_id = std::get<2>(frame_data);

            // Create GStreamer buffer
            GstBuffer *buffer = gst_buffer_new_allocate(NULL, frame.total() * frame.elemSize(), NULL);
            GstMapInfo map;
            if (gst_buffer_map(buffer, &map, GST_MAP_WRITE))
            {
                std::memcpy(map.data, frame.data, frame.total() * frame.elemSize());
                gst_buffer_add_frame_id_meta(buffer, frame_id, context_id);
                gst_buffer_unmap(buffer, &map);
            }

            // Push frame into GStreamer pipeline
            g_signal_emit_by_name(appsrc, "push-buffer", buffer, NULL);
            gst_buffer_unref(buffer);
        }
    }

    void processYOLO()
    {
        while (running)
        {
            std::unique_lock<std::mutex> lock(tensorMutex);
            tensorCond.wait(lock, [this]
                            { return !tensorQueue.empty() || !running; });

            if (!running)
                break;

            auto tensor_data_tuple = std::move(tensorQueue.front());
            tensorQueue.pop();
            lock.unlock();

            std::vector<char> tensor_data = std::move(std::get<0>(tensor_data_tuple));
            guint64 frame_id = std::get<1>(tensor_data_tuple);
            int context_id = std::get<2>(tensor_data_tuple);
            cv::Mat original_frame;
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                auto context_it = frame_storage.find(context_id);
                if (context_it != frame_storage.end())
                {
                    auto frame_it = context_it->second.find(frame_id);
                    if (frame_it != context_it->second.end())
                    {
                        original_frame = std::move(frame_it->second);
                        context_it->second.erase(frame_it);
                    }
                }
            }

            if (!original_frame.empty())
            {
                cv::Mat rgb_image = yolo_post_process(tensor_data, original_frame);
                cv::Mat nv12_frame;
                nv12_frame = convertBGRToNV12(std::move(rgb_image));
                {
                    udpstreamers[context_id]->pushFrame(std::move(nv12_frame));
                }
                std::cout << "Processed YOLOv8 output frame_id: " << frame_id << " context_id: " << context_id << std::endl;
            }
            else
            {
                std::cerr << "Error: Original frame not found for frame ID: " << frame_id << std::endl;
            }
        }
    }

    void stop()
    {
        running = false;
        frameCond.notify_all();
        tensorCond.notify_all();
    }

    ~GStreamerPipeline()
    {
        stop();
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }

private:
    GstElement *pipeline, *appsrc, *appsink, *pciehost, *capsfilter;
    std::queue<std::tuple<std::vector<char>, guint64, int>> tensorQueue;
    std::mutex tensorMutex;
    std::condition_variable tensorCond;
    std::atomic<bool> running{true};

    GstCustomMeta *meta;
    GstStructure *meta_struct;
    static GstFlowReturn onNewSample(GstAppSink *appsink, gpointer user_data)
    {
        GStreamerPipeline *pipeline = static_cast<GStreamerPipeline *>(user_data);
        // Now safely call instance methods
        return pipeline->handleSample(appsink);
    }
    GstFlowReturn handleSample(GstAppSink *appsink)
    {
        GstSample *sample = gst_app_sink_pull_sample(appsink);
        if (!sample)
            return GST_FLOW_ERROR;

        GstBuffer *buffer = gst_sample_get_buffer(sample);
        auto [frame_id, context_id] = gst_buffer_get_frame_id_meta(buffer);
        std::cout << "onNewSample frame_id : " << frame_id << " context_id : " << context_id << std::endl;

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ))
        {
            std::vector<char> tensor_data(map.size);
            std::memcpy(tensor_data.data(), map.data, map.size);
            gst_buffer_unmap(buffer, &map);
            {
                std::lock_guard<std::mutex> lock(tensorMutex);
                tensorQueue.push(std::make_tuple(std::move(tensor_data), frame_id, context_id));
            }
            tensorCond.notify_one();
        }
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
};

void processVideo(const std::string &source, int stream_id)
{
    std::cout << "Starting video processing for source: " << source << " with context ID: " << stream_id << std::endl;
    VideoReader reader(source);

    cv::Mat frame;
    guint64 frame_id = 0;
    while (reader.read(frame))
    {
        cv::Mat nv12_frame = convertBGRToNV12(frame);
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            frame_storage[stream_id][frame_id] = nv12_frame;

            frameQueue.push(std::make_tuple(nv12_frame, frame_id++, stream_id));
            // Prevent memory overflow by removing old frames
            if (frame_storage[stream_id].size() > 100)
            {
                frame_storage[stream_id].erase(frame_storage[stream_id].begin());
            }
        }
        frameCond.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(VIDEO_READ_INTERVAL));
    }
}

void processVideoUSB(const std::string &source, int stream_id)
{
    std::cout << "Starting video processing for source: " << source << " with context ID: " << stream_id << std::endl;
    VideoReaderUSB reader(source);

    cv::Mat frame;
    guint64 frame_id = 0;
    while (reader.read(frame))
    {
        cv::Mat nv12_frame = convertYUYVtoNV12(frame);
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            frame_storage[stream_id][frame_id] = nv12_frame;

            frameQueue.push(std::make_tuple(nv12_frame, frame_id++, stream_id));
            // Prevent memory overflow by removing old frames
            if (frame_storage[stream_id].size() > 100)
            {
                frame_storage[stream_id].erase(frame_storage[stream_id].begin());
            }
        }
        frameCond.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(VIDEO_READ_INTERVAL));
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <video_source1> <video_source2> ..." << std::endl;
        return -1;
    }

    std::vector<std::thread> videoThreads;
    int context_id = 0;
    udpstreamers.reserve(argc - 1);
    for (int i = 1; i < argc; ++i)
    {
        udpstreamers.emplace_back(std::make_unique<VideoStreamer>(context_id));
        context_id++;
    }
    GStreamerPipeline pipeline;

    std::thread gstThread(&GStreamerPipeline::processFrames, &pipeline);
    std::thread yoloThread(&GStreamerPipeline::processYOLO, &pipeline);
    context_id = 0;
    for (int i = 1; i < argc; ++i)
    {
        std::string sourceArgument;
        sourceArgument = argv[i];
        if (sourceArgument.find("rtsp") != std::string::npos)
        {
            videoThreads.emplace_back(processVideo, argv[i], context_id++);
        }
        else
        {
            videoThreads.emplace_back(processVideoUSB, argv[i], context_id++);
        }
    }

    for (auto &thread : videoThreads)
    {
        thread.join();
    }

    pipeline.stop();
    gstThread.join();
    yoloThread.join();

    return 0;
}
