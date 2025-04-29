#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_img.h"



__global__ void kernel_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_y >= t_color_cuda_img.m_size.y || l_x >= t_color_cuda_img.m_size.x) return;

    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x];


    t_bw_cuda_img.m_p_uchar1[l_y * t_bw_cuda_img.m_size.x + l_x].x =
        l_bgr.x * 0.11f + l_bgr.y * 0.59f + l_bgr.z * 0.30f;
}



__global__ void kernel_remove_channel(CudaImg img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img.m_size.x || y >= img.m_size.y) return;

    uchar3& pixel = img.at(y, x);
    int half_w = img.m_size.x / 2;
    int half_h = img.m_size.y / 2;

    if (x < half_w && y < half_h) {
        pixel.x = 0; 
    } else if (x >= half_w && y < half_h) {
        pixel.y = 0; 
    } else if (x < half_w && y >= half_h) {
        pixel.z = 0; 
    } else {
        pixel.x = 0; 
    }
}



__global__ void kernel_split_rotate(CudaImg src, CudaImg dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    uchar3 pixel = src.at(y, x);


    int new_x = y;
    int new_y = src.m_size.x - 1 - x;


    if (new_x < dst.m_size.x && new_y < dst.m_size.y) {
        dst.at(new_y, new_x) = pixel;
    }
}




void cu_run_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size,
                  (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_grayscale<<<l_blocks, l_threads>>>(t_color_cuda_img, t_bw_cuda_img);

  

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}


__global__ void kernel_zesvetli(CudaImg src, CudaImg dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    uchar3 pixel = src.at(y, x);


    const int darkness_decrease = 100; 
    pixel.x = max(pixel.x - darkness_decrease, 0);
    pixel.y = max(pixel.y - darkness_decrease, 0);
    pixel.z = max(pixel.z - darkness_decrease, 0);

    dst.at(y, x) = pixel;
}



void cu_run_brighten(CudaImg src, CudaImg dst) {
    cudaError_t err;
    dim3 block_size(20, 20);
    dim3 grid_size((src.m_size.x + 19) / 20, (src.m_size.y + 19) / 20);

    kernel_zesvetli<<<grid_size, block_size>>>(src,dst);

    if ((err = cudaGetLastError()) != cudaSuccess)
    printf("CUDA Error (split_rotate) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

        cudaDeviceSynchronize();
}

void cu_run_split_rotate(CudaImg src, CudaImg dst)
{
    cudaError_t err;
    dim3 block_size(16, 16);
    dim3 grid_size((src.m_size.x + 15) / 16, (src.m_size.y + 15) / 16);

    kernel_split_rotate<<<grid_size, block_size>>>(src, dst);


    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (split_rotate) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}



void cu_run_remove_channel(CudaImg img)
{
    cudaError_t err;
    dim3 block_size(16, 16);
    dim3 grid_size((img.m_size.x + 15) / 16, (img.m_size.y + 15) / 16);

    kernel_remove_channel<<<grid_size, block_size>>>(img);

   

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (remove_channel) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}



__global__ void kernel_rotate(CudaImg src, CudaImg dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    uchar3 pixel = src.at(y, x);

 
    int new_x = y;
    int new_y = src.m_size.x - 1 - x;

    if (new_x < dst.m_size.x && new_y < dst.m_size.y)
    {
        dst.at(new_y, new_x) = pixel;
    }
}


void cu_run_rotate(CudaImg src, CudaImg dst)
{
    cudaError_t err;
    dim3 block_size(16, 16);
    dim3 grid_size((src.m_size.x + block_size.x - 1) / block_size.x,
                   (src.m_size.y + block_size.y - 1) / block_size.y);

    kernel_rotate<<<grid_size, block_size>>>(src, dst);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (rotate) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}


__global__ void kernel_mirror(CudaImg src, CudaImg dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    uchar3 pixel = src.at(y, x);

 
    int mirrored_x = src.m_size.x - 1 - x;
    int mirrored_y = y;

    if (mirrored_x < dst.m_size.x && mirrored_y < dst.m_size.y)
    {
        dst.at(mirrored_y, mirrored_x) = pixel;
    }
}

void cu_run_mirror(CudaImg src, CudaImg dst)
{
    cudaError_t err;
    dim3 block_size(16, 16);
    dim3 grid_size((src.m_size.x + block_size.x - 1) / block_size.x,
                   (src.m_size.y + block_size.y - 1) / block_size.y);

    kernel_mirror<<<grid_size, block_size>>>(src, dst);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (mirror) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}



__global__ void kernel_ball(CudaImg dst, int ball_x, int ball_y, int ball_radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.m_size.x || y >= dst.m_size.y) return;


    int dx = x - ball_x;
    int dy = y - ball_y;
    if (dx * dx + dy * dy <= ball_radius * ball_radius)
    {
     
        dst.at(y, x) = make_uchar3(0, 255, 0);
    }
    else
    {
   
        dst.at(y, x) = make_uchar3(0, 0, 0);
    }
}

void cu_run_ball(CudaImg dst, int ball_x, int ball_y, int ball_radius)
{
    cudaError_t err;
    dim3 block_size(16, 16);
    dim3 grid_size((dst.m_size.x + block_size.x - 1) / block_size.x,
                   (dst.m_size.y + block_size.y - 1) / block_size.y);

    kernel_ball<<<grid_size, block_size>>>(dst, ball_x, ball_y, ball_radius);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (ball) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}




__global__ void kernel_double_size(CudaImg src, CudaImg dst)
{
    int src_x = blockIdx.x * blockDim.x + threadIdx.x;
    int src_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (src_x >= src.m_size.x || src_y >= src.m_size.y) return;

    uchar3 pixel = src.at(src_y, src_x);

 
    int dst_x = src_x * 2;
    int dst_y = src_y * 2;

   
    if (dst_x + 1 < dst.m_size.x && dst_y + 1 < dst.m_size.y)
    {
        dst.at(dst_y, dst_x) = pixel;
        dst.at(dst_y, dst_x + 1) = pixel;
        dst.at(dst_y + 1, dst_x) = pixel;
        dst.at(dst_y + 1, dst_x + 1) = pixel;
    }
}



void cu_run_double_size(CudaImg src, CudaImg dst)
{
    cudaError_t err;
    dim3 block_size(16, 16);
    dim3 grid_size((src.m_size.x + block_size.x - 1) / block_size.x,
                   (src.m_size.y + block_size.y - 1) / block_size.y);

    kernel_double_size<<<grid_size, block_size>>>(src, dst);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (double size) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}


__global__ void kernel_mirror_halves(CudaImg src, CudaImg dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    int half_width = src.m_size.x / 2;
    uchar3 pixel;

    if (x < half_width) {
   
        int mirrored_x = half_width - 1 - x;
        pixel = src.at(y, mirrored_x);
    } else {
     
        int mirrored_x = src.m_size.x - 1 - (x - half_width);
        pixel = src.at(y, mirrored_x);
    }

    dst.at(y, x) = pixel;
}




void cu_run_mirror_halves(CudaImg src, CudaImg dst)
{
    cudaError_t err;
    dim3 block_size(16, 16);
    dim3 grid_size((src.m_size.x + block_size.x - 1) / block_size.x,
                   (src.m_size.y + block_size.y - 1) / block_size.y);

    kernel_mirror_halves<<<grid_size, block_size>>>(src, dst);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (mirror halves) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}


