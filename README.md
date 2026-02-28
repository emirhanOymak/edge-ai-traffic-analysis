# Real-Time Traffic Analysis & Edge AI System

This repository contains a highly optimized, C++ based Real-Time Object Detection pipeline designed for resource-constrained Edge AI environments. The system utilizes YOLOv8 architecture, shifting away from Python/PyTorch dependencies to achieve hardware-level memory management, deterministic execution, and minimal inference latency.

## Engineering Motivation & Architecture

Prototyping AI models in Python introduces significant overhead due to the Global Interpreter Lock (GIL), JIT compilation delays, and continuous data serialization across APIs. To prove hardware-level optimization capabilities for Embedded Systems, this project was built from scratch in C++ focusing on the following core architectural decisions:

* **Bare-Metal Execution Context:** Designed and tested on a bare-metal Linux environment to eliminate Type-2 Hypervisor context-switching overhead, ensuring lossless access to CPU L1/L2 caches and RAM bandwidth.
* **AOT Compilation & Graph Optimization:** The YOLOv8 model was exported to the universal **ONNX** format. Utilizing the ONNX Runtime C++ API with `ORT_ENABLE_ALL` flags, sequential mathematical operations (e.g., Convolution and Batch Normalization) are fused ahead-of-time (AOT) during memory allocation, significantly reducing memory-bound bottlenecks.
* **Zero-Copy Memory Binding:** Bypassed the standard OpenCV-to-Inference memory duplication overhead. The raw memory pointer of the pre-processed OpenCV `cv::Mat` is bound directly to the ONNX Allocator (`Ort::Value::CreateTensor`). Data flows into the CPU's ALU without intermediate RAM cloning.
* **Spatial Integrity (Letterboxing):** Implemented custom letterboxing during pre-processing to pad the 16:9 video stream into the model's required 640x640 square matrix without aspect-ratio distortion, formatting the memory layout to the SIMD-friendly NCHW planar format.
* **Direct Pointer Arithmetic for Post-Processing:** Instead of taking the transpose of the massive `[1, 84, 8400]` output tensor—which inflates RAM usage—direct stride-based pointer arithmetic is utilized to extract bounding boxes, confidence scores, and class IDs before applying the Non-Maximum Suppression (NMS) algorithm.
* **CPU Core Isolation:** Configured session threads to single-core execution (`SetIntraOpNumThreads(1)`) to simulate deterministic RTOS environments where the remaining CPU cores must be reserved for other critical vehicle sensors and actuators.

## Tech Stack & Dependencies

* **Language:** C++17
* **Build System:** CMake (Cross-platform meta-build system)
* **Computer Vision:** OpenCV 4.x (compiled with DNN module)
* **Inference Engine:** Microsoft ONNX Runtime (C++ API)
* **Model:** YOLOv8-Nano (ONNX format)

## Getting Started

### Prerequisites
Ensure you have the essential build tools and libraries installed on your Ubuntu system:
```bash
sudo apt update && sudo apt install build-essential cmake git pkg-config
sudo apt install libopencv-dev

Note: You must download the ONNX Runtime C++ Linux x64 binaries and extract them to an onnxruntime directory in the project root.

Build Instructions

The project uses CMake to abstract the compilation process. Run the following commands from the root directory:

mkdir build
cd build
cmake ..
make

Execution

Ensure your test video (e.g., traffic.mp4) and the exported ONNX model (yolov8n.onnx) are present in the root directory. Execute the compiled binary:

cd ..
./build/traffic_app

Pipeline Overview

    Pre-processing: Video decoding -> Letterbox Padding -> RGB Swap -> 0-1 Normalization -> NCHW formatting.

    Inference: Zero-copy tensor binding -> ONNX Runtime AOT Graph Execution.

    Post-processing: Pointer arithmetic on the flattened 1D array -> Confidence filtering (>0.25) -> NMS (IoU thresholding) -> Re-scaling bounding boxes to original resolution.