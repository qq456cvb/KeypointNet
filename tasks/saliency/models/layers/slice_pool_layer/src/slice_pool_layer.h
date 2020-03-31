
int slice_pool_max_forward(THFloatTensor *data_tensor, THIntTensor *slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor, THIntTensor *pool_mask_tensor);

int slice_pool_max_backward(THFloatTensor *top_grad_tensor, THIntTensor *pool_mask_tensor, int num_slice, int num_batch, int channels, THFloatTensor *output_tensor );

int slice_pool_avg_forward(THFloatTensor *data_tensor, THIntTensor *slice_idx_tensor, THIntTensor *slice_counts_tensor , int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor);

int slice_pool_avg_backward(THFloatTensor *top_grad_tensor, THIntTensor *slice_idx_tensor, THIntTensor *slice_counts_tensor , int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor );
