// #include "group_points.h"
#include "utils.h"

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
				 const float *points, const int *idx,
				 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
				      int nsample, const float *grad_out,
				      const int *idx, float *grad_points);
                      
int group_points(int b, int c, int n, int npoints, int nsample,
			 at::Tensor points,
			 at::Tensor idx,
			 at::Tensor out) {

    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);

    if (points.is_cuda()) {
        CHECK_CUDA(idx);
    }


    group_points_kernel_wrapper(b, c, n, npoints, nsample, points.data_ptr<float>(), idx.data_ptr<int>(), out.data_ptr<float>());
    return 1;
}

int group_points_grad(int b, int c, int n, int npoints, int nsample,
			      at::Tensor grad_out,
			      at::Tensor idx,
			      at::Tensor grad_points) {
    
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);

    if (grad_out.is_cuda()) {
        CHECK_CUDA(idx);
    } 

    group_points_grad_kernel_wrapper(b, c, n, npoints, nsample, grad_out.data_ptr<float>(), idx.data_ptr<int>(),
				     grad_points.data_ptr<float>());
    return 1;
}
