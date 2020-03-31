int slice_unpool_forward_cuda( THCudaTensor * data_tensor, THCudaIntTensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THCudaTensor * output_tensor );

int slice_unpool_backward_cuda( THCudaTensor * top_grad_tensor, THCudaIntTensor * slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THCudaTensor * output_tensor );

