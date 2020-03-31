#ifndef _S_POOLING_KERNEL
#define _S_POOLING_KERNEL

#ifdef __cplusplus
extern "C" {
#endif
    
    int slice_pool_max_forward_gpu_laucher(float * data, int * slice_idx, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, int * pool_mask, cudaStream_t stream);

    int slice_pool_max_backward_gpu_laucher(float * top, int * pool_mask, const int num_slice, const int num_batch, const int channels, float * output, cudaStream_t stream);

    int slice_pool_avg_forward_gpu_laucher(float * data, int * slice_idx, int * slice_counts, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream);

    int slice_pool_avg_backward_gpu_laucher(float * top, int * slice_idx, int * slice_counts, const int num_slice, const int num_batch, const int channels, const int num_points, float * output, cudaStream_t stream);

    
#ifdef __cplusplus
}
#endif

#endif
