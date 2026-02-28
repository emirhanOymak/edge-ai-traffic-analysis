cmake_minimum_required(VERSION 3.10)
project(EdgeAITraffic)

# Modern C++ özellikleri için C++17 kullanıyoruz
set(CMAKE_CXX_STANDARD 17)

# OpenCV kütüphanesini sistemde bul
find_package(OpenCV REQUIRED)

# ONNX Runtime başlık (header) ve kütüphane dosyalarının yolları
set(ONNXRUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Derlenecek ana dosyamız
add_executable(traffic_app main.cpp)

# OpenCV ve ONNX Runtime'ı çalıştırılabilir dosyamıza bağlıyoruz (linkliyoruz)
target_link_libraries(traffic_app ${OpenCV_LIBS} onnxruntime)