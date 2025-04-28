#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_img.h"

///////////////////////////////////////////////////////////////////////////////
// Kernel: Convert color image to grayscale
///////////////////////////////////////////////////////////////////////////////

__global__ void kernel_grayscale(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_y >= t_color_cuda_img.m_size.y || l_x >= t_color_cuda_img.m_size.x) return;

    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x];

    // Standard formula: 0.11 * B + 0.59 * G + 0.30 * R
    t_bw_cuda_img.m_p_uchar1[l_y * t_bw_cuda_img.m_size.x + l_x].x =
        l_bgr.x * 0.11f + l_bgr.y * 0.59f + l_bgr.z * 0.30f;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel: Remove one color channel per quadrant
///////////////////////////////////////////////////////////////////////////////

__global__ void kernel_remove_channel(CudaImg img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img.m_size.x || y >= img.m_size.y) return;

    uchar3& pixel = img.at(y, x);
    int half_w = img.m_size.x / 2;
    int half_h = img.m_size.y / 2;

    if (x < half_w && y < half_h) {
        pixel.x = 0; // Top-left: remove Blue
    } else if (x >= half_w && y < half_h) {
        pixel.y = 0; // Top-right: remove Green
    } else if (x < half_w && y >= half_h) {
        pixel.z = 0; // Bottom-left: remove Red
    } else {
        pixel.x = 0; // Bottom-right: remove Blue again (optional)
    }
}

///////////////////////////////////////////////////////////////////////////////
// Kernel: Rotate each quadrant by 90 degrees
///////////////////////////////////////////////////////////////////////////////

__global__ void kernel_split_rotate(CudaImg src, CudaImg dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    uchar3 pixel = src.at(y, x);

    // Otočení doprava o 90°
    int new_x = y;
    int new_y = src.m_size.x - 1 - x;

    // POZOR: at(y, x) → první parametr je řádek (tedy new_y), druhý sloupec (new_x)
    if (new_x < dst.m_size.x && new_y < dst.m_size.y) {
        dst.at(new_y, new_x) = pixel;
    }
}




///////////////////////////////////////////////////////////////////////////////
// Wrapper: Calls the grayscale kernel
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Wrapper: Calls the split and rotate kernel
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel_zesvetli(CudaImg src, CudaImg dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.m_size.x || y >= src.m_size.y) return;

    uchar3 pixel = src.at(y, x);

    // Brighten the pixel
    const int brightness_increase = 100; // amount of brightness
    pixel.x = min(pixel.x + brightness_increase, 255);
    pixel.y = min(pixel.y + brightness_increase, 255);
    pixel.z = min(pixel.z + brightness_increase, 255);

    // Store back into the destination image at the same (x, y)
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

///////////////////////////////////////////////////////////////////////////////
// Wrapper: Calls the color removal kernel
///////////////////////////////////////////////////////////////////////////////

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

    // Rotate 90° to the right
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

    // Calculate mirrored X coordinate
    int mirrored_x = src.m_size.x - 1 - x;
    int mirrored_y = y; // Y stays the same

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

    // Vzdálenost od středu míčku
    int dx = x - ball_x;
    int dy = y - ball_y;
    if (dx * dx + dy * dy <= ball_radius * ball_radius)
    {
        // Pokud je pixel uvnitř míčku, nastav ho na zelenou barvu (tenisový míček)
        dst.at(y, x) = make_uchar3(0, 255, 0);
    }
    else
    {
        // Jinak pozadí (černé)
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

