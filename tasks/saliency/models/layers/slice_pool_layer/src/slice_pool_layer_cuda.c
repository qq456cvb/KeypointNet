#include <THC/THC.h>
#include <math.h>
#include "cuda/slice_pool_layer_cuda_kernel.h"

extern THCState *state;

// Max Pooling Forward CUDA
int slice_pool_max_forward_cuda( THCudaTensor * data_tensor, THCudaIntTensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THCudaTensor * output_tensor, THCudaIntTensor * pool_mask_tensor )
{
    // Grab the input tensor
    float * data = THCudaTensor_data(state, data_tensor);
    int * slice_idx = THCudaIntTensor_data(state, slice_idx_tensor);
    float * output = THCudaTensor_data(state, output_tensor);
    int * pool_mask = THCudaIntTensor_data(state, pool_mask_tensor);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_pool_max_forward_gpu_laucher( data, slice_idx, num_slice, num_batch, channels, num_points, output, pool_mask, stream );

    return 1;
    
}



// Max Pooling Backward CUDA
int slice_pool_max_backward_cuda( THCudaTensor * top_grad_tensor, THCudaIntTensor * pool_mask_tensor, int num_slice, int num_batch, int channels, THCudaTensor * output_tensor )
{
    // Grab the input tensor
    float * top_grad = THCudaTensor_data(state, top_grad_tensor);
    int * pool_mask = THCudaIntTensor_data(state, pool_mask_tensor);
    float * output = THCudaTensor_data(state, output_tensor);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_pool_max_backward_gpu_laucher( top_grad, pool_mask, num_slice, num_batch, channels, output, stream );
    
    return 1;
    
}




// Avg Pooling Forward CUDA
int slice_pool_avg_forward_cuda( THCudaTensor * data_tensor, THCudaIntTensor * slice_idx_tensor, THCudaIntTensor * slice_counts_tensor, int num_slice, int num_batch, int channels, int num_points, THCudaTensor * output_tensor)
{
    // Grab the input tensor
    float * data = THCudaTensor_data(state, data_tensor);
    int * slice_idx = THCudaIntTensor_data(state, slice_idx_tensor);
    int * slice_counts = THCudaIntTensor_data(state, slice_counts_tensor);
    float * output = THCudaTensor_data(state, output_tensor);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_pool_avg_forward_gpu_laucher( data, slice_idx, slice_counts, num_slice, num_batch, channels, num_points, output, stream );
    
    return 1;
    
}


// Avg Pooling Backward CUDA
int slice_pool_avg_backward_cuda( THCudaTensor * top_grad_tensor, THCudaIntTensor * slice_idx_tensor, THCudaIntTensor * slice_counts_tensor, int num_slice, int num_batch, int channels, int num_points, THCudaTensor * output_tensor )
{
    // Grab the input tensor
    float * top_grad = THCudaTensor_data(state, top_grad_tensor);
    int * slice_idx = THCudaIntTensor_data(state, slice_idx_tensor);
    int * slice_counts = THCudaIntTensor_data(state, slice_counts_tensor);
    float * output = THCudaTensor_data(state, output_tensor);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_pool_avg_backward_gpu_laucher( top_grad, slice_idx, slice_counts, num_slice, num_batch, channels, num_points, output, stream );
    
    return 1;
    
}


