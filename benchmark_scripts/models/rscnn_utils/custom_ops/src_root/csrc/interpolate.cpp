#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #include "interpolate.h"
#include "utils.h"


void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
			     const float *known, float *dist2, int *idx);

void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
				      const float *points, const int *idx,
				      const float *weight, float *out);

void three_interpolate_grad_kernel_wrapper(int b, int n, int c, int m,
					   const float *grad_out,
					   const int *idx, const float *weight,
					   float *grad_points);

                       
void three_nn(int b, int n, int m, at::Tensor unknown,
		      at::Tensor known, at::Tensor dist2,
		      at::Tensor idx) {
    CHECK_CONTIGUOUS(unknown);
    CHECK_CONTIGUOUS(known);
    CHECK_IS_FLOAT(unknown);
    CHECK_IS_FLOAT(known);

    if (unknown.is_cuda()) {
        CHECK_CUDA(known);
    }

    three_nn_kernel_wrapper(b, n, m, unknown.data_ptr<float>(), known.data_ptr<float>(),
                            dist2.data_ptr<float>(), idx.data_ptr<int>());
}

void three_interpolate(int b, int c, int m, int n,
			       at::Tensor points,
			       at::Tensor idx,
			       at::Tensor weight,
			       at::Tensor out) {

    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    if (points.is_cuda()) {
        CHECK_CUDA(idx);
        CHECK_CUDA(weight);
    }

    three_interpolate_kernel_wrapper(b, c, m, n, points.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(),
        out.data_ptr<float>());
}

void three_interpolate_grad(int b, int c, int n, int m,
				    at::Tensor grad_out,
				    at::Tensor idx,
				    at::Tensor weight,
				    at::Tensor grad_points) {

    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    if (grad_out.is_cuda()) {
        CHECK_CUDA(idx);
        CHECK_CUDA(weight);
    }

    three_interpolate_grad_kernel_wrapper(b, c, n, m, grad_out.data_ptr<float>(), idx.data_ptr<int>(), weight.data_ptr<float>(),
					  grad_points.data_ptr<float>());
}

