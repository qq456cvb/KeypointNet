
// #include "ball_query.h"
#include "utils.h"


void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
				     int nsample, const float *new_xyz,
				     const float *xyz, const int *fps_idx, int *idx);

int ball_query(int b, int n, int m, float radius, int nsample,
		       at::Tensor new_xyz, at::Tensor xyz, at::Tensor fps_idx,
		       at::Tensor idx) {

    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);

    if (new_xyz.is_cuda()) {
        CHECK_CUDA(xyz);
    }

    query_ball_point_kernel_wrapper(b, n, m, radius, nsample, new_xyz.data_ptr<float>(), xyz.data_ptr<float>(), fps_idx.data_ptr<int>(), idx.data_ptr<int>());
    return 1;
}

