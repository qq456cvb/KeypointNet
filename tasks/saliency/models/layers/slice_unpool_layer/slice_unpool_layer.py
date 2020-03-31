import torch
from torch.autograd import Function
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ext_unpool import slice_unpool_layer
from torch.autograd import Variable


class Slice_Unpool(Function):
    def forward(self, input, slice_idx_mat):
        
        assert input.is_contiguous()
        assert slice_idx_mat.is_contiguous()
        
        num_batch, channels, num_slice, _ = input.size()
        num_points = slice_idx_mat.size(1)
        self.num_slice = num_slice
        
        out = torch.zeros(num_batch, channels, num_points, 1)
    
        self.slice_idx_mat = slice_idx_mat
    
        if not input.is_cuda:
            slice_unpool_layer.slice_unpool_forward(input, slice_idx_mat, num_slice, num_batch, channels, num_points, out)
        else:
            out = out.cuda()
            slice_unpool_layer.slice_unpool_forward_cuda(input, slice_idx_mat, num_slice, num_batch, channels, num_points, out)
        
        return out


    def backward(self, grad_top):
        num_slice = self.num_slice
        num_batch, channels, num_points, _ = grad_top.size()
        grad_bottum = torch.zeros(num_batch, channels, num_slice, 1)
        
        
        if not grad_top.is_cuda:
            slice_unpool_layer.slice_unpool_backward(grad_top, self.slice_idx_mat, num_slice, num_batch, channels, num_points, grad_bottum)
        else:
            grad_bottum = grad_bottum.cuda()
            slice_unpool_layer.slice_unpool_backward_cuda(grad_top, self.slice_idx_mat, num_slice, num_batch, channels, num_points, grad_bottum)
            #print grad_bottum.min(), grad_bottum.max(), grad_bottum.size()
        
        return grad_bottum, None


class SU(torch.nn.Module):
    def __init__(self):
        super(SU, self).__init__()
        
        self.su=Slice_Unpool()
    #
    def forward(self, input, slice_idx_mat):
        return self.su(input, slice_idx_mat)
