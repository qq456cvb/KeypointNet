#pragma once
#include <torch/extension.h>

int ball_query(int b, int n, int m, float radius, int nsample,
		       at::Tensor new_xyz, at::Tensor xyz, at::Tensor fps_idx,
		       at::Tensor idx);
