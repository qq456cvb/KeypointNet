// #include "sampling.h"
#include "utils.h"



void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
				  const float *points, const int *idx,
				  float *out);

void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
				       const float *grad_out, const int *idx,
				       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
					    const float *dataset, float *temp,
					    int *idxs);
                        
int gather_points(int b, int c, int n, int npoints,
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

    gather_points_kernel_wrapper(b, c, n, npoints, points.data_ptr<float>(), idx.data_ptr<int>(), out.data_ptr<float>());
    return 1;
}

int gather_points_grad(int b, int c, int n, int npoints,
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

    gather_points_grad_kernel_wrapper(b, c, n, npoints, grad_out.data_ptr<float>(), idx.data_ptr<int>(),
				      grad_points.data_ptr<float>());
    return 1;
}

int furthest_point_sampling(int b, int n, int m,
				    at::Tensor points,
				    at::Tensor temp,
				    at::Tensor idx) {

    CHECK_CONTIGUOUS(points);
    CHECK_IS_FLOAT(points);

    furthest_point_sampling_kernel_wrapper(b, n, m, points.data_ptr<float>(), temp.data_ptr<float>(), idx.data_ptr<int>());
    return 1;
}
