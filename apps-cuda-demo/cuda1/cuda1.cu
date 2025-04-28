#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

// Kernel: Displays thread hierarchy and linear index
__global__ void thread_hierarchy()
{
    // Global variables
    int l_x = threadIdx.x + blockIdx.x * blockDim.x;
    int l_y = threadIdx.y + blockIdx.y * blockDim.y;

    // Compute grid width in number of threads
    int grid_width = gridDim.x * blockDim.x;

    // Compute linear index
    int linear_index = l_y * grid_width + l_x;

    printf("Block{%d,%d}[%d,%d] Thread{%d,%d}[%d,%d] Global[%d,%d] LinearIndex=%d\n",
           gridDim.x, gridDim.y, blockIdx.x, blockIdx.y,
           blockDim.x, blockDim.y, threadIdx.x, threadIdx.y,
           l_x, l_y, linear_index);
}

void cu_run_cuda(dim3 t_grid_size, dim3 t_block_size)
{
    cudaError_t l_cerr;

    // Launch kernel
    thread_hierarchy<<< t_grid_size, t_block_size>>>();

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    // Synchronize to flush printf output
    cudaDeviceSynchronize();
}