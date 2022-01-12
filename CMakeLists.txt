cmake_minimum_required(VERSION 3.2)
project(yolo_reid)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/leo/LibProjects/TensorRT-7.2.2.3/include/)
link_directories(/home/leo/LibProjects/TensorRT-7.2.2.3/lib/)
#opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)

cuda_add_library(yolov5_model SHARED
        src/yolo/yolov5_model.cpp
        src/yolo/calibrator.cpp
        src/yolo/preprocess.cu
        src/yolo/common.cpp
        src/yolo/yololayer.cu)
target_link_libraries(yolov5_model nvinfer)
target_link_libraries(yolov5_model cudart)
target_link_libraries(yolov5_model ${OpenCV_LIBS})

cuda_add_library(fastrt_model
        src/fastrt/reid_model.cpp
        src/fastrt/baseline.cpp
        src/fastrt/embedding_head.cpp
        src/fastrt/factory.cpp
        src/fastrt/InferenceEngine.cpp
        src/fastrt/layers.cpp
        src/fastrt/model.cpp
        src/fastrt/poolingLayerRT.cpp
        src/fastrt/sbs_resnet.cpp
        src/fastrt/utils.cpp)
target_link_libraries(fastrt_model nvinfer)
target_link_libraries(fastrt_model cudart)
target_link_libraries(fastrt_model ${OpenCV_LIBS})

cuda_add_executable(yolo_reid main.cpp)
target_link_libraries(yolo_reid yolov5_model fastrt_model)