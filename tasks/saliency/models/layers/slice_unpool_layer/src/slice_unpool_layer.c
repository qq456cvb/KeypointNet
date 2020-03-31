
#include <TH/TH.h>
#include <math.h>
#include <stdio.h>


//-------- Max Unpooling Functions

// Max Pooling Forward function
int slice_unpool_forward(THFloatTensor *data_tensor, THIntTensor *slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor)
{
    // slice_idx:               num_batch, num_points
    // data:                    num_batch, channels, num_slice  , 1
    // output:                  num_batch, channels, num_points, 1
    
    float * data = THFloatTensor_data(data_tensor);
    float * output = THFloatTensor_data(output_tensor);
    int * slice_idx = THIntTensor_data(slice_idx_tensor);
    
    // Max Unpooling
    int n, c, i;
    for (n = 0; n < num_batch; n++) {
        for (i = 0; i < num_points; i++) {
            for (c = 0; c < channels; c++) {
                
                // get slice index
                int cls_idx = slice_idx[ n*num_points + i ];
                
                // get output index, [n, c, i, 0]
                int output_idx =  n * channels * num_points + c * num_points + i; // output[n, c, ori_idx, 0]
                
                // get input index, [n,, c, cls_idx, 0]
                int input_index =  n * channels * num_slice + c * num_slice + cls_idx; // data[n, c, m, 0]
                
                output[ output_idx ] = data[input_index];
                
            }
        }
    }
    return 1;
}




// Max unpooling Backward function
int slice_unpool_backward(THFloatTensor *top_grad_tensor, THIntTensor *slice_idx_tensor, int num_slice, int num_batch, int channels, int num_points, THFloatTensor *output_tensor )
{
    // top grad:    num_batch, channels, num_points, 1
    // slice_idx:   num_batch, num_points
    // output:      num_batch, channels, num_slice, 1
    
    float * top_grad = THFloatTensor_data(top_grad_tensor);
    int * slice_idx = THIntTensor_data(slice_idx_tensor);
    float * output = THFloatTensor_data(output_tensor);
    
    int n, c, i;
    for (n = 0; n < num_batch; n++) {
        for (i = 0; i < num_points; i++) {
            for (c = 0; c < channels; c++) {
            
                
                // get slice index
                int s_idx = slice_idx[ n*num_points + i ];
                
                
                int top_index = n * channels * num_points + c * num_points + i; //top[n, c, i, 0]
                
                int bottom_index = n * channels * num_slice + c * num_slice + s_idx ; // output[n, c, s_idx, 0]
                
                output[bottom_index] += top_grad[top_index];
                
            }
        }
    }
    return 1;
}






