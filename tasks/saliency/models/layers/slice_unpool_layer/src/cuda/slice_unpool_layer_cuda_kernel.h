#ifndef _S_POOLING_KERNEL
#define _S_POOLING_KERNEL

#ifdef __cplusplus
extern "C" {
#endif
    
    int slice_unpool_forward_gpu_laucher(float * data, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream);

    int slice_unpool_backward_gpu_laucher(float * top, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream);

    
    
#ifdef __cplusplus
}
#endif

#endif
