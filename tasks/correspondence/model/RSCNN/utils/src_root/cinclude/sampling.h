#pragma once
#include <torch/extension.h>

int gather_points(int b, int c, int n, int npoints,
			  at::Tensor points,
			  at::Tensor idx,
			  at::Tensor out);

int gather_points_grad(int b, int c, int n, int npoints,
			       at::Tensor grad_out,
			       at::Tensor idx,
			       at::Tensor grad_points);

int furthest_point_sampling(int b, int n, int m,
				    at::Tensor points,
				    at::Tensor temp,
				    at::Tensor idx);
