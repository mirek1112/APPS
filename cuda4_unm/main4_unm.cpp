
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {}

// CUDA function declarations
void cu_run_grayscale(CudaImg src, CudaImg dst);
void cu_run_split_rotate(CudaImg src, CudaImg dst);
void cu_run_remove_channel(CudaImg img);
void cu_run_brighten(CudaImg src, CudaImg dst);
void cu_run_rotate(CudaImg src, CudaImg dst);
void cu_run_mirror(CudaImg src, CudaImg dst);

void cu_run_double_size(CudaImg src, CudaImg dst);
void cu_run_mirror_halves(CudaImg src, CudaImg dst);


void cu_run_ball(CudaImg dst, int ball_x, int ball_y, int ball_radius);
void run_ball_animation();

int main(int argc, char **argv)
{

    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (argc < 2)
    {
        printf("Enter picture filename!\n");
        return 1;
    }

    cv::Mat color_cv_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (!color_cv_img.data)
    {
        printf("Unable to read file '%s'\n", argv[1]);
        return 1;
    }

    int width = color_cv_img.cols;
    int height = color_cv_img.rows;

    cv::Mat grayscale_cv_img(color_cv_img.size(), CV_8U);
    cv::Mat rotated_cv_img(cv::Size(height, width), CV_8UC3); 
    cv::Mat brightened_cv_img(color_cv_img.size(), CV_8UC3);
    cv::Mat rotated90_cv_img(cv::Size(height, width), CV_8UC3);
    cv::Mat mirrored_cv_img(color_cv_img.size(), CV_8UC3);
    cv::Mat doubled_cv_img(cv::Size(width * 2, height * 2), CV_8UC3); 


    cv::Mat mirrored_half_cv_img(color_cv_img.size(), CV_8UC3);

 
    CudaImg color_cuda_img, 
    grayscale_cuda_img, 
    rotated_cuda_img, 
    brightened_cuda_img, 
    rotated90_cuda_img,
    mirrored_cuda_img,
    doubled_cuda_img,
    mirrored_half_cuda_img;


    color_cuda_img.m_size = make_uint3(width, height, 1);
    grayscale_cuda_img.m_size = make_uint3(width, height, 1);
    rotated_cuda_img.m_size = make_uint3(height, width, 1);
    brightened_cuda_img.m_size = make_uint3(width, height, 1);
    rotated90_cuda_img.m_size = make_uint3(height, width, 1);
    mirrored_cuda_img.m_size = make_uint3(width, height, 1);
    doubled_cuda_img.m_size = make_uint3(width * 2, height * 2, 1);
    mirrored_half_cuda_img.m_size = make_uint3(width, height, 1);



    color_cuda_img.m_p_uchar3 = (uchar3 *)color_cv_img.data;
    grayscale_cuda_img.m_p_uchar1 = (uchar1 *)grayscale_cv_img.data;
    rotated_cuda_img.m_p_uchar3 = (uchar3 *)rotated_cv_img.data;
    brightened_cuda_img.m_p_uchar3 = (uchar3 *)brightened_cv_img.data;
    rotated90_cuda_img.m_p_uchar3 = (uchar3 *)rotated90_cv_img.data;
    mirrored_cuda_img.m_p_uchar3 = (uchar3 *)mirrored_cv_img.data;
    doubled_cuda_img.m_p_uchar3 = (uchar3 *)doubled_cv_img.data;
    mirrored_half_cuda_img.m_p_uchar3 = (uchar3 *)mirrored_cv_img.data;




    cu_run_grayscale(color_cuda_img, grayscale_cuda_img);
    cu_run_split_rotate(color_cuda_img, rotated_cuda_img);
    cu_run_remove_channel(rotated_cuda_img);
    cu_run_brighten(color_cuda_img, brightened_cuda_img);
    cu_run_rotate(color_cuda_img, rotated90_cuda_img);
    cu_run_mirror(color_cuda_img, mirrored_cuda_img);
    cu_run_double_size(color_cuda_img, doubled_cuda_img);
    cu_run_mirror_halves(color_cuda_img, mirrored_cuda_img);


    cv::imshow("Original Color", color_cv_img);
    cv::imshow("Grayscale", grayscale_cv_img);
    cv::imshow("Rotated + Channel Removed", rotated_cv_img);
    cv::imshow("Brightened", brightened_cv_img);
    cv::imshow("Rotated 90 Degrees", rotated90_cv_img);
    cv::imshow("Mirrored", mirrored_cv_img);
    cv::imshow("Double Size", doubled_cv_img);
    cv::imshow("Zrcadlo Na pul", mirrored_cv_img);
  
    run_ball_animation();

    cv::waitKey(0);

    return 0;
}



void run_ball_animation()
{
 
    cv::Mat cv_img(cv::Size(640, 480), CV_8UC3);

    CudaImg cuda_img;
    cuda_img.m_size = make_uint3(640, 480, 1);
    cuda_img.m_p_uchar3 = (uchar3 *)cv_img.data;

    int ball_radius = 20;
    int ball_x = 320;
    int ball_y = 240;
    int speed_y = 5;
    int direction = 1;

    while (true)
    {
    
        ball_y += direction * speed_y;

        if (ball_y >= 480 - ball_radius || ball_y <= ball_radius)
            direction *= -1; 

      
        cu_run_ball(cuda_img, ball_x, ball_y, ball_radius);

      
        cv::imshow("Ball Animation", cv_img);
        if (cv::waitKey(30) >= 0) break; 
    }
}
