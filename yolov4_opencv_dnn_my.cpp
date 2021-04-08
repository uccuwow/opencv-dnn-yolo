#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;

std::vector<std::string> class_names;
// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);


void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    char buff[20];
    snprintf(buff, sizeof(buff), "%.2f", conf);
    std::string label = buff;
    if (!class_names.empty())
    {
        CV_Assert(classId < (int)class_names.size());
        label = class_names[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = std::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - labelSize.height),
              cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}

int main()
{
    {
        std::ifstream class_file("model/coco.names");
        if (!class_file)
        {
            std::cerr << "failed to open names file\n";
            return 0;
        }

        std::string line;
        while (std::getline(class_file, line))
            class_names.push_back(line);
    }

    cv::VideoCapture source("demo.mp4");

    auto net = cv::dnn::readNetFromDarknet("model/yolov4.cfg", "model/yolov4.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;
    cv::namedWindow("output");
    while(cv::waitKey(1) < 1)
    {
        source >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }
        
        double total_start = (double)cv::getTickCount();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        double dnn_start = (double)cv::getTickCount();
        net.forward(detections, output_names);
        double dnn_end = (double)cv::getTickCount();

        std::vector<int> classId;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (auto& output : detections)
        {
            float* data = (float*)output.data;
            for (int j = 0; j < output.rows; ++j, data += output.cols)
            {
                cv::Mat scores = output.row(j).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.70)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classId.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.70, 0.45, indices);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            drawPred(classId[idx], confidences[idx], box.x, box.y,
                     box.x + box.width, box.y + box.height, frame);
        }

        double total_end = (double)cv::getTickCount();

        double inference_ms = ((double)dnn_end - dnn_start) / cv::getTickFrequency() * 1000;
        double total_ms = ((double)total_end - total_start) / cv::getTickFrequency() * 1000;
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference ms: " << inference_ms << ", Total ms: " << total_ms;
        std::cout << stats_ss.str() << std::endl;
        auto stats = stats_ss.str();
            
        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

        
        cv::imshow("output", frame);
    }

    return 0;
}
