#pragma once
#include <torch/extension.h>

int group_points(int b, int c, int n, int npoints, int nsample,
			 at::Tensor points,
			 at::Tensor idx, at::Tensor out);
             
int group_points_grad(int b, int c, int n, int npoints, int nsample,
			      at::Tensor grad_out,
			      at::Tensor idx,
			      at::Tensor grad_points);
