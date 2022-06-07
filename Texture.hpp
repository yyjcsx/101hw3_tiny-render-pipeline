//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
        std::cout<<width<<"   "<<height<<std::endl;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        //???为啥把这个注掉就能跑了？
        //一定要跟一遍代码，看看uv到底从头到尾咋变的，以及这里为啥不行
        // if(u<0) u = 0;
        // if(u>1) u = 1;
        // if(v<0) v = 0;


        
        // if(v>0) v = 1;
        //std::cout<<u<<" "<<v<<std::endl;
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

};
#endif //RASTERIZER_TEXTURE_H
