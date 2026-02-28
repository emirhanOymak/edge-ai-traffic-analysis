#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        std::cout << "Initializing C++ Edge AI System..." << std::endl;

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8_Inference");
        Ort::SessionOptions session_options;
        
        // Set to 1 for edge scenarios, 0 to use all available cores
        session_options.SetIntraOpNumThreads(0); 
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const char* model_path = "yolov8n.onnx";
        Ort::Session session(env, model_path, session_options);

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};

        // --- DATASET INTEGRATION ---
        // Change "traffic.mp4" to your new downloaded dataset video name
        cv::VideoCapture cap("traffic.mp4");
        if (!cap.isOpened()) {
            std::cerr << "Error: Video file not found! Please check the path." << std::endl;
            return -1;
        }

        cv::Mat frame, blob;
        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // COCO class names (Filtered for traffic-related objects in English)
        std::vector<std::string> class_names = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"};

        while (cap.read(frame)) {
            auto total_start = std::chrono::high_resolution_clock::now();

            // --- 1. EDGE AI PRE-PROCESSING: LETTERBOX ---
            int max_dim = std::max(frame.cols, frame.rows);
            cv::Mat square_frame(max_dim, max_dim, CV_8UC3, cv::Scalar(114, 114, 114));
            frame.copyTo(square_frame(cv::Rect(0, 0, frame.cols, frame.rows)));

            float factor = max_dim / 640.0f;

            cv::dnn::blobFromImage(square_frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

            // --- 2. INFERENCE ---
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

            // --- 3. POST-PROCESSING ---
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            
            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;

            for (int i = 0; i < 8400; ++i) {
                float max_score = 0.0f;
                int best_class_id = -1;

                for (int c = 0; c < 80; ++c) {
                    float score = output_data[(4 + c) * 8400 + i];
                    if (score > max_score) {
                        max_score = score;
                        best_class_id = c;
                    }
                }

                // Confidence Threshold set to 0.25
                if (max_score > 0.25f) {
                    float x = output_data[0 * 8400 + i];
                    float y = output_data[1 * 8400 + i];
                    float w = output_data[2 * 8400 + i];
                    float h = output_data[3 * 8400 + i];

                    int left = int((x - 0.5 * w) * factor);
                    int top = int((y - 0.5 * h) * factor);
                    int width = int(w * factor);
                    int height = int(h * factor);

                    confidences.push_back(max_score);
                    class_ids.push_back(best_class_id);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }

            // Non-Maximum Suppression (NMS)
            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, nms_result);

            // Drawing Bounding Boxes
            for (int idx : nms_result) {
                cv::Rect box = boxes[idx];
                int class_id = class_ids[idx];
                
                if (class_id < class_names.size()) {
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
                    std::string label = class_names[class_id] + ": " + std::to_string(confidences[idx]).substr(0, 4);
                    cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                }
            }

            auto total_end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
            
            cv::putText(frame, "FPS: " + std::to_string(1000.0 / total_ms).substr(0, 5), 
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            cv::imshow("Edge AI Real-Time Traffic Analysis", frame);

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

    } catch (const Ort::Exception& exception) {
        std::cerr << "ONNX Runtime Error: " << exception.what() << std::endl;
        return -1;
    }

    return 0;
}