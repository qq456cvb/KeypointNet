
#include <TH/TH.h>
#include <math.h>
#include <stdio.h>


//-------- Max Pooling Functions

// Max Pooling Forward function
int slice_pool_max_forward(THFloatTensor *data_tensor, THIntTensor *slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor, THIntTensor *pool_mask_tensor)
{
    // data:                    num_batch, channels,  num_points  , 1
    // output:                  num_batch, channels,  num_slice, 1
    // slice_idx:               num_batch, num_points
    
    float * data = THFloatTensor_data(data_tensor);
    int * slice_idx = THIntTensor_data(slice_idx_tensor);
    float * output = THFloatTensor_data(output_tensor);
    int * pool_mask = THIntTensor_data(pool_mask_tensor);
    
    // Max Pooling
    int n, c, m;
    for (n = 0; n < num_batch; n++) {
        for (c = 0; c < channels; c++) {
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
    return 1;
}




// Max Pooling Backward function
int slice_pool_max_backward(THFloatTensor *top_grad_tensor, THIntTensor *pool_mask_tensor, int num_slice, int num_batch, int channels, THFloatTensor *output_tensor )
{
    // top grad:    num_batch, channels, slice, 1
    // pool mask:   num_batch, channels, slice, 1
    // output:      num_batch, channels, num_points, 1
    
    float * top_grad = THFloatTensor_data(top_grad_tensor);
    int * pool_mask = THIntTensor_data(pool_mask_tensor);
    float * output = THFloatTensor_data(output_tensor);
    
    int n, c, m;
    for (n = 0; n < num_batch; n++) {
        for (c = 0; c < channels; c++) {
            for (m = 0; m < num_slice; m++) {
                
                int top_index = n * channels * num_slice + c * num_slice + m; // output[n,c,m,0]
                int bottom_index = pool_mask[top_index];
                
                if (bottom_index != -1) {
                    output[bottom_index] += top_grad[top_index];
                }
                
            }
        }
    }
    return 1;
}




//-------- Avg Pooling Functions

// Average Pooling Forward function
int slice_pool_avg_forward(THFloatTensor *data_tensor, THIntTensor *slice_idx_tensor, THIntTensor *slice_counts_tensor , int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor)
{
    // data:            num_batch, channels, num_points    , 1
    // output:          num_batch, channels, slice  , 1
    // slice_idx:       num_batch, num_points
    // slice_counts:    num_batch, slice
    
    float * data = THFloatTensor_data(data_tensor);
    int * slice_idx = THIntTensor_data(slice_idx_tensor);
    int * slice_counts = THIntTensor_data(slice_counts_tensor);
    float * output = THFloatTensor_data(output_tensor);
    
    
    // Max Pooling
    int n,c,m;
    for (n = 0; n < num_batch; n++) {
        for (c = 0; c < channels; c++) {
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
    return 1;
}



// Avg Pooling Backward function
int slice_pool_avg_backward(THFloatTensor *top_grad_tensor, THIntTensor *slice_idx_tensor, THIntTensor *slice_counts_tensor , int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor )
{
    // top grad:        num_batch, num_slice,  1
    // slice_idx:       num_batch, num_points
    // slice_counts:    num_batch, slice
    // output:          num_batch, channels, num_points, 1
    
    float * top_grad = THFloatTensor_data(top_grad_tensor);
    int * slice_idx = THIntTensor_data(slice_idx_tensor);
    int * slice_counts = THIntTensor_data(slice_counts_tensor);
    float * output = THFloatTensor_data(output_tensor);
    
    int n, c, m;
    for (n = 0; n < num_batch; n++) {
        for (c = 0; c < channels; c++) {
            for (m = 0; m < num_points; m++) {
                
                // get slice index
                int c_idx = slice_idx[ n * num_points + m ] ; // slice_idx[n, m]
                
                // get slice counts idx
                float slice_count = slice_counts[ n * num_slice + c_idx ]; // slice_counts[n, c_idx]
                
                // get top idx
                int top_idx = n * channels * num_slice + c * num_slice + c_idx; // output[n, c, c_idx]

                // get bottom idx
                int bottom_idx = n * channels * num_points + c * num_points + m; // data[n, c, m, 0]
                
                if (bottom_idx != -1) {
                    output[bottom_idx] += top_grad[top_idx] / slice_count;
                }
            }
        }
    }
    return 1;
}







