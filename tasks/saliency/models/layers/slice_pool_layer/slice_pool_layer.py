import torch
from torch.autograd import Function
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ext_pool import slice_pool_layer
from torch.autograd import Variable


class Slice_Pool(Function):
    def __init__(self, num_slice):
        super().__init__()
        self.num_slice = num_slice
    
    def forward(self, input, slice_idx_mat, pool_type, slice_counts):
        
        assert input.is_contiguous()
        
        if pool_type.numpy()[0] == 1:
            self.pool_type = "Max_Pool"
        elif pool_type.numpy()[0] == 2:
            self.pool_type = "Avg_Pool"
            
        num_batch, channels, num_points, _ = input.size()
        num_slice = self.num_slice
        if num_slice == None:
            num_slice = int(slice_idx_mat.max()) + 1
            
        self.num_points = num_points
        
        if self.pool_type == "Max_Pool":
            out = torch.ones(num_batch, channels, num_slice, 1) * 0
        else:
            out = torch.zeros(num_batch, channels, num_slice, 1)
    
        pool_mask = torch.IntTensor(num_batch, channels, num_slice, 1)
        pool_mask[:,:,:,:] = -1

        # Max_Pool
        if self.pool_type == "Max_Pool":
            if not input.is_cuda:
                slice_pool_layer.slice_pool_max_forward(input, slice_idx_mat, num_slice, num_batch, channels, num_points, out, pool_mask)
                self.pool_mask = pool_mask
            else:
                out = out.cuda()
                pool_mask = pool_mask.cuda()
                slice_pool_layer.slice_pool_max_forward_cuda(input, slice_idx_mat, num_slice, num_batch, channels, num_points, out, pool_mask)

            self.pool_mask = pool_mask


        # Avg_Pool
        elif self.pool_type == "Avg_Pool":
            self.slice_idx_mat = slice_idx_mat
            self.slice_counts = slice_counts
            if not input.is_cuda:
                slice_pool_layer.slice_pool_avg_forward(input, slice_idx_mat, slice_counts, num_slice, num_batch, channels, num_points, out)
            else:
                out = out.cuda()
                slice_pool_layer.slice_pool_avg_forward_cuda(input, slice_idx_mat, slice_counts, num_slice, num_batch, channels, num_points, out)
            
        
        return out


    def backward(self, grad_top):
        num_batch, channels, num_slice, _ = grad_top.size()
        grad_bottum = torch.zeros(num_batch, channels, self.num_points, 1).cuda()
        # Max Pooling
        if self.pool_type == "Max_Pool":
            if not grad_top.is_cuda:
                slice_pool_layer.slice_pool_max_backward(grad_top, self.pool_mask, num_slice, num_batch, channels, grad_bottum)
            else:
                # grad_bottum = grad_bottum.cuda()
                slice_pool_layer.slice_pool_max_backward_cuda(grad_top, self.pool_mask, num_slice, num_batch, channels, grad_bottum)


    
        # Avg_Pool
        elif self.pool_type == "Avg_Pool":
            if not grad_top.is_cuda:
                slice_pool_layer.slice_pool_avg_backward(grad_top, self.slice_idx_mat, self.slice_counts, num_slice, num_batch, channels, self.num_points, grad_bottum)
            else:
                grad_bottum = grad_bottum.cuda()
                slice_pool_layer.slice_pool_avg_backward_cuda(grad_top, self.slice_idx_mat, self.slice_counts, num_slice, num_batch, channels, self.num_points, grad_bottum)
                
            
        return grad_bottum, None, None, None





class SP(torch.nn.Module):
    def __init__(self, pool_type, num_slice=None):
        super(SP, self).__init__()
        if pool_type == "Max_Pool":
            self.pool_type = Variable(torch.ones(1,1)) * 1
        elif pool_type == "Avg_Pool":
            self.pool_type = Variable(torch.ones(1,1)) * 2
        
        self.sp=Slice_Pool(num_slice)
    #
    def forward(self, input, slice_idx_mat, slice_counts=torch.autograd.Variable(None)):
        return self.sp(input, slice_idx_mat, self.pool_type, slice_counts)


