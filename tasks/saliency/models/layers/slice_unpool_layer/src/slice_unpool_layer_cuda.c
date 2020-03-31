#include <THC/THC.h>
#include <math.h>
#include "cuda/slice_unpool_layer_cuda_kernel.h"

extern THCState *state;

// Max Unpooling Forward CUDA
int slice_unpool_forward_cuda( THCudaTensor * data_tensor, THCudaIntTensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THCudaTensor * output_tensor )
{
    // Grab the input tensor
    float * data = THCudaTensor_data(state, data_tensor);
    float * output = THCudaTensor_data(state, output_tensor);
    int * slice_idx = THCudaIntTensor_data(state, slice_idx_tensor);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_unpool_forward_gpu_laucher( data, slice_idx, num_slice, num_batch, channels, num_points, output, stream );

    return 1;
    
}



// Max Unpooling Backward CUDA
int slice_unpool_backward_cuda( THCudaTensor * top_grad_tensor, THCudaIntTensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THCudaTensor * output_tensor )
{
    // Grab the input tensor
    float * top_grad = THCudaTensor_data(state, top_grad_tensor);
    int * slice_idx = THCudaIntTensor_data(state, slice_idx_tensor);
    float * output = THCudaTensor_data(state, output_tensor);
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    
    slice_unpool_backward_gpu_laucher( top_grad, slice_idx, num_slice, num_batch, channels, num_points, output, stream );
    
    return 1;
    
}



