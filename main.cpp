#include "yolo/yolov5_model.h"
#include "fastrt/reid_model.h"

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {

    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
        if (net.size() == 2 && net[1] == '6') {
            is_p6 = true;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Yolov5_Model yolov5model;
    yolov5model.deserialize_engine("../yolov5s_int8.engine");
    Reid_model reidmodel;
    cv::Mat img;
    cv::Mat resimg;
    std::vector<cv::Rect> rects;
    cv::VideoCapture cap("../paobu2/VID_20220107_155023.mp4");
    cv::VideoWriter writer("../yolo_reid_demo2.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)), true);
//    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
//    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    for (int i=0;i<cap.get(cv::CAP_PROP_FRAME_COUNT);i++) {
        cap.read(img);
        rects.clear();
        auto res = yolov5model.run_frame(img);
        for (auto re: res) {
            if (re.class_id == 0) {
                cv::Rect r = get_rect(img, re.bbox);
//                std::cout<<r.x<<" "<<r.width<<" "<<r.y<<" "<<r.height<< std::endl;
                rects.emplace_back(r);
            }
        }
        auto reid_ress = reidmodel.run_frame(img, rects);

        for (size_t i = 0; i < res.size(); i++) {
            if (reid_ress[i].conf>0.95)
            {
                cv::putText(img, std::to_string(reid_ress[i].id), cv::Point(rects[i].x, rects[i].y - 1), cv::FONT_HERSHEY_PLAIN,
                            3, cv::Scalar(255, 192, 0), 2);
            }
            else
            {
                cv::putText(img, "?", cv::Point(rects[i].x, rects[i].y - 1), cv::FONT_HERSHEY_PLAIN,
                            3, cv::Scalar(255, 192, 0), 2);
            }

        }
        cv::resize(img, resimg, cv::Size(0,0),0.5,0.5);
        cv::imshow("img", img);
        writer.write(img);
        cv::waitKey(1);
    }
    cap.release();
    writer.release();
    return 0;
}