#ifdef __cplusplus
extern "C" {
#endif
    
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "slice_pool_layer_cuda_kernel.h"
    
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)


    
// -------- Max Pooling Forward kernel

// nthreads: n * c
__global__ void slice_pool_max_forward_gpu(const int nthreads, float * data, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, int * pool_mask)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads)
    {
        
        // get the index of point
        const int c = index % channels;
        const int n = index / channels;

        // Max Pooling
        int m;
        for (m = 0; m < num_points; m++) {
            
            // get slice index
            int idx = slice_idx[n * num_points + m];   // slice_idx[n, m]
            
            // get output index
            int output_idx =  n * channels * num_slice + c * num_slice + idx; // output[n, c, idx, 0]
            
            // get pool mask index
            int pool_mask_idx =  n * channels * num_slice + c * num_slice + idx; // output[n, c, idx, 0]
            
            // get input index
            int input_index =  n * channels * num_points + c * num_points + m; // data[n, c, m, 0]
            
            
            if (data[input_index] > output[ output_idx ])
            {
                output[output_idx] = data[input_index];
                pool_mask[pool_mask_idx] = input_index;
            }
            
            
        }
        

    }
}


// -------- Max Pooling Backward kernel
__global__ void slice_pool_max_backward_gpu(const int nthreads, float * top, int * pool_mask, const int num_slice, const int num_batch, const int channels, float * output)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads)
    {
        // get the index of point
        const int c = index % channels;
        const int n = index / channels;
        
        int m;
        for (m = 0; m < num_slice; m++) {
            int top_index = n * channels * num_slice + c * num_slice + m; // output[n,c,m,0]
            int bottom_index = pool_mask[top_index];
            if (bottom_index != -1) {
//                atomicAdd(&output[bottom_index], top[top_index]);
                output[bottom_index] += top[top_index];
            }
        }
    }
}



// -------- Avg Pooling Forward kernel
__global__ void slice_pool_avg_forward_gpu(const int nthreads, float * data, int * slice_idx, int * slice_counts, const int num_slice, const int num_batch, const int channels, const int num_points, float * output)
{
    
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // get the index of point
        const int c = index / num_batch;
        const int n = index % num_batch;

        // Average Pooling
        int m;
        for (m = 0; m < num_points; m++) {

            // get slice idx
            int c_idx = slice_idx[ n * num_points + m ] ; // slice_idx[n, m]
            
            // get slice counts idx
            float slice_count = slice_counts[ n * num_slice + c_idx ]; // slice_counts[n, c_idx]
            
            // get output idx
            int output_idx = n * channels * num_slice + c * num_slice + c_idx; // output[n, c, c_idx]
            
            // get input idx
            int input_idx = n * channels * num_points + c * num_points + m; // data[n, c, m, 0]
            
            output[ output_idx ] += data[input_idx] / slice_count;
            
        }
    }
}

    
    
    
// -------- Avg Pooling Backward kernel

__global__ void slice_pool_avg_backward_gpu(const int nthreads, float * top, int * slice_idx, int * slice_counts, const int num_slice, const int num_batch, const int channels, const int num_points, float * output)
{
    
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // get the index of point
        const int c = index / num_batch;
        const int n = index % num_batch;
        
        int m;
        for (m = 0; m < num_points; m++) {
            
            // get slice index
            int s_idx = slice_idx[ n * num_points + m ] ; // slice_idx[n, m]
            
            // get slice counts idx
            float slice_count = slice_counts[ n * num_slice + s_idx ]; // slice_counts[n, s_idx]
            
            // get top idx
            int top_idx = n * channels * num_slice + c * num_slice + s_idx; // output[n, c, s_idx]
            
            // get bottom idx
            int bottom_idx = n * channels * num_points + c * num_points + m; // data[n, c, m, 0]
            if (bottom_idx != -1) {
                output[bottom_idx] += top[top_idx] / slice_count;
            }
        }
    }
}

    
    
    
// -------- Max Pooling Forward laucher
int slice_pool_max_forward_gpu_laucher(float * data, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, int * pool_mask, cudaStream_t stream)

{
    
    const int kThreadsPerBlock = 1024;
    const int kBlocks = (num_batch * channels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cudaError_t err;
    
    
    slice_pool_max_forward_gpu<<< kBlocks, kThreadsPerBlock, 0, stream>>>(num_batch * channels, data, slice_idx, num_slice, num_batch, channels, num_points, output, pool_mask);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    return 1;
}



// -------- Max Pooling Backward laucher
int slice_pool_max_backward_gpu_laucher(float * top, int * pool_mask, const int num_slice, const int num_batch, const int channels, float * output, cudaStream_t stream)

{
    const int kThreadsPerBlock = 1024;
    const int kBlocks = (num_batch * channels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cudaError_t err;
    
    //const int pooled_num_points = num_points / 2;
    
    slice_pool_max_backward_gpu<<< kBlocks, kThreadsPerBlock, 0, stream>>>(num_batch * channels, top, pool_mask, num_slice, num_batch, channels, output);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    return 1;
}
    

    
// -------- Avg Pooling Forward laucher
int slice_pool_avg_forward_gpu_laucher(float * data, int * slice_idx, int * slice_counts, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream)

{
    const int kThreadsPerBlock = 1024;
    const int kBlocks = (num_batch * channels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cudaError_t err;

    slice_pool_avg_forward_gpu<<< kBlocks, kThreadsPerBlock, 0, stream>>>(num_batch * channels, data, slice_idx, slice_counts, num_slice, num_batch, channels, num_points, output);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    return 1;
}

    
// -------- Avg Pooling Backward laucher
int slice_pool_avg_backward_gpu_laucher(float * top, int * slice_idx, int * slice_counts, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream)

{
    const int kThreadsPerBlock = 1024;
    const int kBlocks = (num_batch * channels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    cudaError_t err;
    
    
    slice_pool_avg_backward_gpu<<< kBlocks, kThreadsPerBlock, 0, stream>>>(num_batch * channels, top, slice_idx, slice_counts, num_slice, num_batch, channels, num_points, output);
    
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
