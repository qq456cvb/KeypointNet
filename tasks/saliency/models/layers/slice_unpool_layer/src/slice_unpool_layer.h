
int slice_unpool_forward(THFloatTensor *data_tensor, THIntTensor *slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor);


int slice_unpool_backward(THFloatTensor *top_grad_tensor, THIntTensor *slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor );

