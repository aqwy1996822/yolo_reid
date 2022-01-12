//
// Created by leo on 2021/10/26.
//

#include "fastrt/reid_model.h"

Reid_model::Reid_model() {
    load_engine();
    load_galley();
}
std::vector<reid_res> Reid_model::run_frame(cv::Mat img, std::vector<cv::Rect> rects)
{
    std::vector<cv::Mat> input;
    std::vector<reid_res> reid_ress;

    for (int i=0;i<rects.size()-REID_MAX_BATCH_SIZE;i+=REID_MAX_BATCH_SIZE) {
        if (i>=rects.size())
        {
            break;
        }

        input.clear();
        TimePoint start_infer, end1_infer, end_infer;

        for (int j = 0; j < REID_MAX_BATCH_SIZE; j++) {
            cv::Mat resizeImg(modelCfg.input_h, modelCfg.input_w, CV_8UC3);

            if (j<rects.size()-i*REID_MAX_BATCH_SIZE)
            {
                cv::resize(img(rects[i + j]), resizeImg, resizeImg.size(), 0, 0, cv::INTER_CUBIC);
            }
            input.emplace_back(resizeImg);
        }
        start_infer = Time::now();
        /* run inference */
        baseline.inference(input);
        float *feat_embedding = baseline.getOutput();
        end1_infer = Time::now();

        T1 = Eigen::Map<Eigen::Matrix<float, REID_OUTPUT_SIZE, REID_MAX_BATCH_SIZE>>(feat_embedding, REID_OUTPUT_SIZE,
                                                                                     REID_MAX_BATCH_SIZE).transpose();
        for (int l = 0; l < T1.rows(); l++) {
            T1.row(l).normalize();
        }
        dist = T1 * T2;
        for (int k = 0; k < REID_MAX_BATCH_SIZE; k++) {
            if (k>=rects.size()-i*REID_MAX_BATCH_SIZE)
            {
                break;
            }
            int maxRow, maxCol;
            dist.block(k, 0, 1, REID_MAX_BATCH_SIZE).maxCoeff(&maxRow, &maxCol);
            std::cout << i + k << " col " << maxCol <<"num"<<dist.block(k, 0, 1, REID_MAX_BATCH_SIZE).row(maxRow).col(maxCol)<< std::endl;
            reid_res res={dist.block(k, 0, 1, REID_MAX_BATCH_SIZE)(maxRow,maxCol), maxCol};
            reid_ress.emplace_back(res);
        }
    }
    return reid_ress;
}
void Reid_model::run_files() {
    std::vector<cv::Mat> input;
    auto filenames = io::fileGlob("../paobu/*.jpg");

    for (int i=0;i<547-REID_MAX_BATCH_SIZE;i+=REID_MAX_BATCH_SIZE)
    {

        input.clear();
        TimePoint start_infer, end1_infer, end_infer;

        for (int j=0;j<REID_MAX_BATCH_SIZE;j++)
        {
            cv::Mat resizeImg(modelCfg.input_h, modelCfg.input_w, CV_8UC3);
            cv::resize(cv::imread(filenames[i+j]), resizeImg, resizeImg.size(), 0, 0, cv::INTER_CUBIC);

//                cv::imshow("img", resizeImg);
//                cv::waitKey(10);
            input.emplace_back(resizeImg);
        }
        start_infer = Time::now();
        /* run inference */
        baseline.inference(input);
        float* feat_embedding = baseline.getOutput();
        end1_infer = Time::now();

        T1 = Eigen::Map<Eigen::Matrix<float,REID_OUTPUT_SIZE,REID_MAX_BATCH_SIZE>>(feat_embedding,REID_OUTPUT_SIZE,REID_MAX_BATCH_SIZE).transpose();
        for (int l =0;l<T1.rows();l++)
        {
            T1.row(l).normalize();
        }
        dist = T1*T2;
        for (int k =0;k<REID_MAX_BATCH_SIZE;k++)
        {
            int maxRow, maxCol;
            dist.block(k, 0, 1, REID_MAX_BATCH_SIZE).maxCoeff(&maxRow,&maxCol);
            std::cout<<filenames[i+k]<<" col "<<maxCol<<std::endl;
        }
//        std::cout << dist << std::endl;
        end_infer = Time::now();
        std::cout << "[Preprocessing+Inference]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1_infer - start_infer).count() << "ms" << std::endl;
        std::cout << "[Posprocessing]: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - end1_infer).count() << "ms" << std::endl;
    }
}

void Reid_model::wts2engine() {
    ModuleFactory moduleFactory;
    std::cout << "[Serializling Engine]" << std::endl;
    if (!baseline.serializeEngine(ENGINE_PATH,
                                  {std::move(moduleFactory.createBackbone(reidCfg)),
                                   std::move(moduleFactory.createHead(reidCfg))})) {
        std::cout << "SerializeEngine Failed." << std::endl;
    }
}

void Reid_model::load_engine() {
    std::cout << "[Deserializling Engine]" << std::endl;
    if(!baseline.deserializeEngine(ENGINE_PATH)) {
        std::cout << "DeserializeEngine Failed." << std::endl;
    }
}

void Reid_model::load_galley() {
    std::vector<std::string> filegalle={
            "songfeng_0.jpg",
            "zhenhu_0.jpg",
            "lingan_0.jpg",
            "hanlin_0.jpg",
            "zhiqin_0.jpg",
            "jinhang_0.jpg"
    };
    std::vector<cv::Mat> gallery;

    for (int i=0;i<REID_MAX_BATCH_SIZE;i++) {
        cv::Mat resizeImg(modelCfg.input_h, modelCfg.input_w, CV_8UC3);
        if (i<filegalle.size())
        {
            cv::resize(cv::imread("../paobu2/"+filegalle[i]), resizeImg, resizeImg.size(), 0, 0, cv::INTER_CUBIC);
        }
        gallery.emplace_back(resizeImg);
    }
    cv::destroyAllWindows();

    /* run inference */
    baseline.inference(gallery);
    float* gallery_embedding = baseline.getOutput();
    const int gallery_size=(int)filegalle.size();
    T2 = Eigen::Map<Eigen::Matrix<float,REID_OUTPUT_SIZE,REID_MAX_BATCH_SIZE>>(gallery_embedding,REID_OUTPUT_SIZE,REID_MAX_BATCH_SIZE);
//        T2.block(0, MAX_BATCH_SIZE, OUTPUT_SIZE, MAX_BATCH_SIZE) = Eigen::Map<Eigen::Matrix<float,OUTPUT_SIZE,MAX_BATCH_SIZE>>(gallery_embedding,OUTPUT_SIZE,MAX_BATCH_SIZE);
    for (int l =0;l<T2.cols();l++)
    {
        T2.col(l).normalize();
    }
}