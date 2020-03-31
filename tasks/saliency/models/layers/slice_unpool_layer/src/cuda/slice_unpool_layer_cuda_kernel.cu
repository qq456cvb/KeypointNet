#ifdef __cplusplus
extern "C" {
#endif
    
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "slice_unpool_layer_cuda_kernel.h"
    
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)


    
// -------- Max Unpooling Forward kernel
// slice_idx:               num_batch, num_points
// data:                    num_batch, channels, num_slice  , 1
// output:                  num_batch, channels, num_points, 1
    
__global__ void slice_unpool_forward_gpu(const int nthreads, float * data, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads)
    {
        
        // get the index of point
        const int c = index / num_points;
        const int i = index % num_points;

        // UnPooling
        int n;
        for (n = 0; n < num_batch; n++) {
            
            // get slice index
            int s_idx = slice_idx[ n*num_points + i ];
            
            // get output index, [n, c, i, 0]
            int output_idx =  n * channels * num_points + c * num_points + i;
            
            // get input index, [n,, c, cls_idx, 0]
            int input_index =  n * channels * num_slice + c * num_slice + s_idx;
            
            output[ output_idx ] = data[input_index];
            
        }
        

    }
}

    
    

// -------- Max Unpooling Backward kernel
__global__ void slice_unpool_backward_gpu(const int nthreads, float * top, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads)
    {
        // get the index of point
        const int c = index / num_batch;
        const int n = index % num_batch;
        
        int i;
        for (i = 0; i < num_points; i++) {

            int s_idx = slice_idx[ n*num_points + i ];
            
            
            int top_index = n * channels * num_points + c * num_points + i; //top[n, c, i, 0]
            
            int bottom_index = n * channels * num_slice + c * num_slice + s_idx ; // output[n, c, cls_idx, 0]
            
            output[bottom_index] += top[top_index];
            
            
        }
        
        
    }
}

    
// -------- Unpooling Forward laucher
int slice_unpool_forward_gpu_laucher(float * data, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream)

{
    
    const int kThreadsPerBlock = 1024;
    const int kBlocks = (num_points * channels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cudaError_t err;
    
    slice_unpool_forward_gpu<<< kBlocks, kThreadsPerBlock, 0, stream>>>(num_points * channels, data, slice_idx, num_slice, num_batch, channels, num_points, output);
    
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    return 1;
}



// -------- Unpooling Backward laucher
int slice_unpool_backward_gpu_laucher(float * top, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream)

{
    const int kThreadsPerBlock = 1024;
    const int kBlocks = (num_batch * channels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cudaError_t err;
    
    slice_unpool_backward_gpu<<< kBlocks, kThreadsPerBlock, 0, stream>>>(num_batch * channels, top, slice_idx, num_slice, num_batch, channels, num_points, output);
    
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    return 1;
}
    
    
    

#ifdef __cplusplus
}
#endif

