// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

// Function prototype from .cu file
void cu_create_chessboard(CudaImg chessboard_img, int square_size);
void cu_create_alphaimg(CudaImg alpha_img, uchar3 color);
void cu_insertimage(CudaImg background_img, CudaImg overlay_img, int2 position, int rotation, int alpha);
void cu_create_uniform_alphaimg(CudaImg img, uchar3 fill_color, unsigned char alpha = 255);
void cu_set_alpha(CudaImg src_bgr, CudaImg dst_bgra, int alpha_override = -1);

void cu_combine_side_by_side(CudaImg left, CudaImg right, CudaImg overlay, CudaImg out);
void cu_resize_nearest(CudaImg src, CudaImg dst);


int main(int argc, char **argv)
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    // Create chessboard background image
    cv::Mat chessboard_cv_mat(511, 515, CV_8UC3);
    CudaImg chessboard_cuda_img;
    chessboard_cuda_img.m_size.x = chessboard_cv_mat.cols;
    chessboard_cuda_img.m_size.y = chessboard_cv_mat.rows;
    chessboard_cuda_img.m_p_uchar3 = (uchar3 *)chessboard_cv_mat.data;

    cu_create_chessboard(chessboard_cuda_img, 21);
    cv::imshow("Chess Board", chessboard_cv_mat);

    // Create alpha image overlay (red square)
    cv::Mat alpha_cv_mat(211, 191, CV_8UC4);
    CudaImg alpha_cuda_img;
    alpha_cuda_img.m_size.x = alpha_cv_mat.cols;
    alpha_cuda_img.m_size.y = alpha_cv_mat.rows;
    alpha_cuda_img.m_p_uchar4 = (uchar4 *)alpha_cv_mat.data;

    cu_create_alphaimg(alpha_cuda_img, {0, 0, 255});
    cv::imshow("Alpha channel", alpha_cv_mat);

  
    cu_insertimage(chessboard_cuda_img, alpha_cuda_img, {11, 23},0,-1);
    cv::imshow("Result I", chessboard_cv_mat);



    cv::Mat left = cv::imread("left.png", cv::IMREAD_COLOR);
    cv::Mat right = cv::imread("left.png", cv::IMREAD_COLOR);
    cv::Mat overlay = cv::imread("ball.png", cv::IMREAD_UNCHANGED);
    cv::Mat combined(left.rows, left.cols + right.cols, CV_8UC3);

    if (!left.data || !right.data) {
        printf("Could not load 'left.jpg' or 'right.jpg' from current directory.\n");
    }

    CudaImg cuda_left = { make_uint3(left.cols, left.rows, 1), (uchar3*)left.data };
    CudaImg cuda_right = { make_uint3(right.cols, right.rows, 1), (uchar3*)right.data };
    CudaImg cuda_combined = { make_uint3(combined.cols, combined.rows, 1), (uchar3*)combined.data };
    CudaImg cuda_overlay;
    cuda_overlay.m_size.x = overlay.cols;
    cuda_overlay.m_size.y = overlay.rows;
    cuda_overlay.m_p_uchar4 = (uchar4 *)overlay.data;




    cu_combine_side_by_side(cuda_left, cuda_right, cuda_overlay, cuda_combined);
    cu_insertimage(cuda_combined, cuda_overlay, {100, 50}, 270,128);
    cv::imshow("Combined", combined);

    cv::Mat resized(combined.rows * 2, combined.cols * 2, CV_8UC3);
    CudaImg cuda_resized = { make_uint3(resized.cols, resized.rows, 1), (uchar3*)resized.data };
    cu_resize_nearest(cuda_combined, cuda_resized);
    cv::imshow("Resized", resized);


    cv::Mat alpha_img(200, 200, CV_8UC4);
    CudaImg alpha_cuda = { make_uint3(alpha_img.cols, alpha_img.rows, 1), (uchar4*)alpha_img.data };


    cu_create_uniform_alphaimg(alpha_cuda, {0, 0, 255}, 125);
    cv::imshow("Uniform", alpha_img);


 
  
    cv::Mat input_bgr = cv::imread("ball.png", cv::IMREAD_COLOR);  // ensures 3 channels
        if (input_bgr.empty()) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

        cv::Mat output_bgra(input_bgr.size(), CV_8UC4);  // target with alpha

        // CUDA structures
        CudaImg cuda_in, cuda_out;

        cuda_in.m_size = make_uint3(input_bgr.cols, input_bgr.rows, 1);
        cuda_in.m_p_uchar3 = (uchar3*)input_bgr.data;

        cuda_out.m_size = make_uint3(output_bgra.cols, output_bgra.rows, 1);
        cuda_out.m_p_uchar4 = (uchar4*)output_bgra.data;

        // Set alpha
        cu_set_alpha(cuda_in, cuda_out, 127);

        // Show result
        cv::imshow("Alpha Set", output_bgra);
        cv::imwrite("output_with_alpha.png", output_bgra);



        cv::Mat newTest(input_bgr.size(), CV_8UC3);  // target with alpha
        CudaImg cuda_newtest;
        cuda_newtest.m_size = make_uint3(newTest.rows, newTest.cols, -1);
        cuda_newtest.m_p_uchar3 = (uchar3*)newTest.data;

        cu_insertimage(cuda_newtest,cuda_out,{0,0},0,-1);

        cv::imshow("Alpha ", newTest);





    
    

    



    // Load user image with alpha if provided
    if (argc > 1)
    {
        cv::Mat user_alpha_img = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

        if (!user_alpha_img.data)
            printf("Unable to read file '%s'\n", argv[1]);
        else if (user_alpha_img.channels() != 4)
            printf("Image does not contain alpha channel!\n");
        else
        {
            CudaImg user_alpha_cuda_img;
            user_alpha_cuda_img.m_size.x = user_alpha_img.cols;
            user_alpha_cuda_img.m_size.y = user_alpha_img.rows;
            user_alpha_cuda_img.m_p_uchar4 = (uchar4 *)user_alpha_img.data;

            int2 center_position = {
                (int)(chessboard_cuda_img.m_size.x / 3),
                (int)(chessboard_cuda_img.m_size.y /3)
            };

            cu_insertimage(chessboard_cuda_img, user_alpha_cuda_img, center_position,90,128);
            cv::imshow("Result II", chessboard_cv_mat);
        }
    }

    cv::waitKey(0);
    return 0;
}
