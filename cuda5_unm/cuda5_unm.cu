// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
// Image transformation from RGB to BW schema. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Demo kernel to create chess board
__global__ void kernel_chessboard(CudaImg img)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (y >= img.m_size.y || x >= img.m_size.x) return;

    unsigned char color = 255 * ((blockIdx.x + blockIdx.y) & 1);
    img.at(y, x) = { color, color, color };
}

void cu_create_chessboard(CudaImg img, int square_size)
{
    cudaError_t err;

    dim3 grid((img.m_size.x + square_size - 1) / square_size,
              (img.m_size.y + square_size - 1) / square_size);
    dim3 block(square_size, square_size);
    kernel_chessboard<<<grid, block>>>(img);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

// Demo kernel to create picture with alpha channel gradient
__global__ void kernel_alphaimg(CudaImg img, uchar3 color)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (y >= img.m_size.y || x >= img.m_size.x) return;

    int diagonal = sqrtf(img.m_size.x * img.m_size.x + img.m_size.y * img.m_size.y);
    int dx = x - img.m_size.x / 2;
    int dy = y - img.m_size.y / 2;
    int dist = sqrtf(dx * dx + dy * dy) - diagonal / 2;

    img.at<uchar4>(y, x) = {color.x, color.y, color.z,(unsigned char)(255 - 255 * dist / (diagonal / 2))
    };
}

void cu_create_alphaimg(CudaImg img, uchar3 fill_color)
{
    cudaError_t err;

    int block_size = 32;
    dim3 grid((img.m_size.x + block_size - 1) / block_size,
              (img.m_size.y + block_size - 1) / block_size);
    dim3 block(block_size, block_size);
    kernel_alphaimg<<<grid, block>>>(img, fill_color);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

// Demo kernel to insert image with alpha blending
__global__ void kernel_insertimage(CudaImg background, CudaImg overlay, int2 position, int rotation, int alpha_override)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= overlay.m_size.x || y >= overlay.m_size.y) return;

    int src_x = x;
    int src_y = y;

    // Otočení overlay obrázku
    if (rotation == 90) {
        src_x = overlay.m_size.y - 1 - y;
        src_y = x;
    } else if (rotation == 180) {
        src_x = overlay.m_size.x - 1 - x;
        src_y = overlay.m_size.y - 1 - y;
    } else if (rotation == 270) {
        src_x = y;
        src_y = overlay.m_size.x - 1 - x;
    }

    int dst_x = src_x + position.x;
    int dst_y = src_y + position.y;

    if (dst_x < 0 || dst_x >= background.m_size.x || dst_y < 0 || dst_y >= background.m_size.y) return;

    uchar4 fg = overlay.at<uchar4>(y, x);
    uchar3 bg = background.at(dst_y, dst_x);

    // Override alpha if needed
    int alpha = (alpha_override >= 0 && alpha_override <= 255) ? alpha_override : fg.w;

    uchar3 blended;
    blended.x = fg.x * alpha / 255 + bg.x * (255 - alpha) / 255;
    blended.y = fg.y * alpha / 255 + bg.y * (255 - alpha) / 255;
    blended.z = fg.z * alpha / 255 + bg.z * (255 - alpha) / 255;

    background.at(dst_y, dst_x) = blended;
}


void cu_insertimage(CudaImg background, CudaImg overlay, int2 position, int rotation = 0, int alpha_override = -1)
{
    cudaError_t err;

    int block_size = 32;
    dim3 grid((overlay.m_size.x + block_size - 1) / block_size,
              (overlay.m_size.y + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    kernel_insertimage<<<grid, block>>>(background, overlay, position, rotation, alpha_override);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (insertimage) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

















__global__ void kernel_combine_side_by_side(CudaImg left, CudaImg right, CudaImg overlay, CudaImg out)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= out.m_size.y || x >= out.m_size.x) return;

    if (x < left.m_size.x && y < left.m_size.y) {
        out.at(y, x) = left.at(y, x);
    } else if (x >= left.m_size.x && x < out.m_size.x && y < right.m_size.y) {
        int rx = x - left.m_size.x;
        out.at(y, x) = right.at(y, rx);
    }
}

void cu_combine_side_by_side(CudaImg left, CudaImg right, CudaImg overlay, CudaImg out)
{
    cudaError_t err;
    dim3 block(32, 32);
    dim3 grid((out.m_size.x + block.x - 1) / block.x,
              (out.m_size.y + block.y - 1) / block.y);

    kernel_combine_side_by_side<<<grid, block>>>(left, right, overlay, out);
    

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (combine) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}



// Kernel to resize an image using nearest-neighbor
__global__ void kernel_resize_nearest(CudaImg src, CudaImg dst)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= dst.m_size.y || x >= dst.m_size.x) return;

    float scale_x = (float)src.m_size.x / dst.m_size.x;
    float scale_y = (float)src.m_size.y / dst.m_size.y;

    int src_x = min((int)(x * scale_x), src.m_size.x - 1);
    int src_y = min((int)(y * scale_y), src.m_size.y - 1);

    dst.at(y, x) = src.at(src_y, src_x);
}

void cu_resize_nearest(CudaImg src, CudaImg dst)
{
    cudaError_t err;
    dim3 block(32, 32);
    dim3 grid((dst.m_size.x + block.x - 1) / block.x,
              (dst.m_size.y + block.y - 1) / block.y);

    kernel_resize_nearest<<<grid, block>>>(src, dst);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (resize) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}



__global__ void kernel_set_alpha(CudaImg src_bgr, CudaImg dst_bgra, int alpha_override)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src_bgr.m_size.x || y >= src_bgr.m_size.y) return;

    uchar3 color = src_bgr.at(y, x);

    unsigned char alpha = (alpha_override >= 0 && alpha_override <= 255) ? alpha_override : 255;

    dst_bgra.at<uchar4>(y, x) = { color.x, color.y, color.z, alpha };
}


void cu_set_alpha(CudaImg src_bgr, CudaImg dst_bgra, int alpha_override)
{
    dim3 block(32, 32);
    dim3 grid((src_bgr.m_size.x + block.x - 1) / block.x,
              (src_bgr.m_size.y + block.y - 1) / block.y);

    kernel_set_alpha<<<grid, block>>>(src_bgr, dst_bgra, alpha_override);

    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [set_alpha] - '%s'\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
}



















__global__ void kernel_create_uniform_alpha(CudaImg img, uchar3 color, unsigned char alpha)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (y >= img.m_size.y || x >= img.m_size.x) return;

    img.at<uchar4>(y, x) = { color.x, color.y, color.z, alpha };
}


void cu_create_uniform_alphaimg(CudaImg img, uchar3 fill_color, unsigned char alpha = 255)
{
    cudaError_t err;

    int block_size = 32;
    dim3 grid((img.m_size.x + block_size - 1) / block_size,
              (img.m_size.y + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    kernel_create_uniform_alpha<<<grid, block>>>(img, fill_color, alpha);

    if ((err = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error (cu_create_uniform_alphaimg) [%d] - '%s'\n", __LINE__, cudaGetErrorString(err));

    cudaDeviceSynchronize();
}