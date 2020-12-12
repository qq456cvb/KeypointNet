#pragma once

#include <torch/extension.h>

void three_nn(int b, int n, int m, at::Tensor unknown,
		      at::Tensor known, at::Tensor dist2,
		      at::Tensor idx);
			  
void three_interpolate(int b, int c, int m, int n,
			       at::Tensor points,
			       at::Tensor idx,
			       at::Tensor weight,
			       at::Tensor out);

void three_interpolate_grad(int b, int c, int n, int m,
				    at::Tensor grad_out,
				    at::Tensor idx,
				    at::Tensor weight,
				    at::Tensor grad_points);
