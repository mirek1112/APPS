#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>


// Kernel to print array elements orderly by using only thread 0 in each block
__global__ void kernel_rozsireni(int *pole, int t_length)
{
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if (inx < t_length)
    {
        char binary_string[9];  
        binary_string[8] = '\0';

        for (int i = 7; i >= 0; --i)
        {
            binary_string[7 - i] = ((pole[inx] >> i) & 1) ? '1' : '0';
        }

        printf("%3d -> %s\n", pole[inx], binary_string);
    }
}

void cu_run_mult(float *t_array, int t_length, float t_mult)
{
    cudaError_t l_cerr;
    int l_threads = 128;
    int l_blocks = (t_length + l_threads - 1) / l_threads;

    // Allocate Unified Memory for the array
    int *pole_0_255;
    cudaMallocManaged(&pole_0_255, 255 * sizeof(int));

    // Fill array on CPU side
    for (int i = 0; i < 255; i++)
    {
        pole_0_255[i] = i;
    }

   

    // Calculate blocks and threads for printing kernel
    int bloky = 2;
    int vlakna = 128;

    // Launch ordered printing kernel
    kernel_rozsireni<<<1, 255>>>(pole_0_255, 255);

    // Synchronize to ensure all kernels finish
    cudaDeviceSynchronize();

    // Check for any CUDA errors
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    // Free Unified Memory
    cudaFree(pole_0_255);
}