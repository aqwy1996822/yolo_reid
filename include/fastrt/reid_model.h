//
// Created by leo on 2021/10/26.
//

#ifndef FASTRT0_0_5_REID_MODEL_H
#define FASTRT0_0_5_REID_MODEL_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "fastrt/utils.h"
#include "fastrt/baseline.h"
#include "fastrt/factory.h"

#define REID_OUTPUT_SIZE 2048
#define REID_MAX_BATCH_SIZE 8

using namespace fastrt;
using namespace nvinfer1;

struct reid_res{
    float conf;
    int id;
};

class Reid_model {
public:
    Reid_model();
    void run_files();
    std::vector<reid_res> run_frame(cv::Mat img, std::vector<cv::Rect> rects);
    void load_galley();
    void load_engine();
    void wts2engine();
private:
    /* Ex1. sbs_R50-ibn */
    const std::string WEIGHTS_PATH = "../marksbs_R50i.wts";
    const std::string ENGINE_PATH = "../marksbs_R50i.engine";


    const int INPUT_H = 384;
    const int INPUT_W = 128;
    const int DEVICE_ID = 0;

    const FastreidBackboneType BACKBONE = FastreidBackboneType::r50;
    const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
    const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
    const int LAST_STRIDE = 1;
    const bool WITH_IBNA = true;
    const bool WITH_NL = true;
    const int EMBEDDING_DIM = 0;

    trt::ModelConfig modelCfg {
            WEIGHTS_PATH,
            REID_MAX_BATCH_SIZE,
            INPUT_H,
            INPUT_W,
            REID_OUTPUT_SIZE,
            DEVICE_ID};

    FastreidConfig reidCfg {
            BACKBONE,
            HEAD,
            HEAD_POOLING,
            LAST_STRIDE,
            WITH_IBNA,
            WITH_NL,
            EMBEDDING_DIM};

    Baseline baseline{modelCfg};

    Eigen::Matrix<float,REID_MAX_BATCH_SIZE,REID_OUTPUT_SIZE> T1;
    Eigen::Matrix<float,REID_OUTPUT_SIZE,REID_MAX_BATCH_SIZE> T2;
    Eigen::Matrix<float, REID_MAX_BATCH_SIZE, REID_MAX_BATCH_SIZE> dist;

};


#endif //FASTRT0_0_5_REID_MODEL_H
