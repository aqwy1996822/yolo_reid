//
// Created by leo on 2022/1/5.
//

#ifndef YOLOV5_MODEL_YOLOV5_MODEL_H
#define YOLOV5_MODEL_YOLOV5_MODEL_H


#include <iostream>
#include <chrono>
#include <cmath>
#include "yolo/cuda_utils.h"
#include "yolo/logging.h"
#include "yolo/common.h"
#include "yolo/utils.h"
#include "yolo/calibrator.h"
#include "yolo/preprocess.h"

class Yolov5_Model {
public:
    Logger gLogger;
    //设置推理数据类型
    bool int8_flag = false;
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.6
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3840 * 2160 // ensure it exceed the maximum size in the input images !

    const char *INPUT_BLOB_NAME = "data";
    const char *OUTPUT_BLOB_NAME = "prob";
    const int INPUT_H = Yolo::INPUT_H;
    static const int INPUT_W = Yolo::INPUT_W;
    static const int CLASS_NUM = Yolo::CLASS_NUM;
    static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) +
                                   1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

    float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;

    float *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex;
    int outputIndex;
    uint8_t *img_host;
    uint8_t *img_device;
    // Create stream
    cudaStream_t stream;
    //控制解析时是否需要释放内存
    bool predict_flag = false;

    Yolov5_Model();

    ~Yolov5_Model();

    int serialize_engine(std::string wts_name, std::string engine_name, bool is_p6, float gd, float gw);

    int deserialize_engine(std::string engine_name);

    int get_width(int x, float gw, int divisor);

    int get_depth(int x, float gd);

    ICudaEngine *
    build_engine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt, float &gd,
                 float &gw, std::string &wts_name);

    ICudaEngine *
    build_engine_p6(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt, float &gd,
                    float &gw, std::string &wts_name);

    void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, bool &is_p6, float &gd, float &gw,
                    std::string &wts_name);

    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchSize);

    std::vector<Yolo::Detection> run_frame(cv::Mat img);

    int run_frames(std::vector<cv::Mat> imgs_buffer);

    int run_files(std::string img_dir);

    int build(std::string wts_name, std::string engine_name, bool is_p6, float gd, float gw, std::string img_dir);
};


#endif //YOLOV5_MODEL_YOLOV5_MODEL_H
