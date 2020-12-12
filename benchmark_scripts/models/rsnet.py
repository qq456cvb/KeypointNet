import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


#-- slice processing utils
def gen_slice_idx(data, resolution, axis=2):
    indices = np.zeros((  data.shape[0], data.shape[2] ))
    for n in range(data.shape[0]):
        indices[n] = gen_slice_idx_routine( data[n], resolution, axis )
    #
    return indices


def gen_slice_idx_routine(data, resolution, axis):
    z_min, z_max = data[:,:,axis].min(), data[:,:,axis].max()

    #gap = (z_max - z_min + 0.001) / numSlices
    gap = resolution
    indices = np.ones( ( data.shape[1], 1 ) ) * float('inf')
    for i in range( data.shape[1]  ):
        z = data[0,i,axis]
        idx = int( (z - z_min) / gap )
        indices[i, 0] = idx
    return indices[:, 0]


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RSNet(nn.Module):
    def __init__(self, cfg):
        super(RSNet, self).__init__()

        resolution_true = [cfg.network.res, cfg.network.res, cfg.network.res]

        # - modified resolution for easy indexing
        self.resolution = [i + 0.00001 for i in resolution_true]
        num_slice = [0, 0, 0]
        num_slice[0] = int(cfg.network.rg / self.resolution[0]) + 1
        num_slice[1] = int(cfg.network.rg / self.resolution[1]) + 1
        num_slice[2] = int(cfg.network.rg / self.resolution[2]) + 1
        self.num_slice = num_slice
        # input: B, 1, N, 3

        # -- conv block 1
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1))
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn_2 = nn.BatchNorm2d(64)

        self.conv_3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn_3 = nn.BatchNorm2d(64)

        # -- RNN block
        num_slice_x, num_slice_y, num_slice_z = num_slice

        self.rnn_type = 'GRU'
        self.rnn_hidden_sz_list = [256, 128, 64, 64, 128, 256]

        self.rnn_x_1 = nn.GRU(64, self.rnn_hidden_sz_list[0], 1, bidirectional=True)
        self.rnn_x_2 = nn.GRU(512, self.rnn_hidden_sz_list[1], 1, bidirectional=True)
        self.rnn_x_3 = nn.GRU(256, self.rnn_hidden_sz_list[2], 1, bidirectional=True)
        self.rnn_x_4 = nn.GRU(128, self.rnn_hidden_sz_list[3], 1, bidirectional=True)
        self.rnn_x_5 = nn.GRU(128, self.rnn_hidden_sz_list[4], 1, bidirectional=True)
        self.rnn_x_6 = nn.GRU(256, self.rnn_hidden_sz_list[5], 1, bidirectional=True)

        self.rnn_y_1 = nn.GRU(64, self.rnn_hidden_sz_list[0], 1, bidirectional=True)
        self.rnn_y_2 = nn.GRU(512, self.rnn_hidden_sz_list[1], 1, bidirectional=True)
        self.rnn_y_3 = nn.GRU(256, self.rnn_hidden_sz_list[2], 1, bidirectional=True)
        self.rnn_y_4 = nn.GRU(128, self.rnn_hidden_sz_list[3], 1, bidirectional=True)
        self.rnn_y_5 = nn.GRU(128, self.rnn_hidden_sz_list[4], 1, bidirectional=True)
        self.rnn_y_6 = nn.GRU(256, self.rnn_hidden_sz_list[5], 1, bidirectional=True)

        self.rnn_z_1 = nn.GRU(64, self.rnn_hidden_sz_list[0], 1, bidirectional=True)
        self.rnn_z_2 = nn.GRU(512, self.rnn_hidden_sz_list[1], 1, bidirectional=True)
        self.rnn_z_3 = nn.GRU(256, self.rnn_hidden_sz_list[2], 1, bidirectional=True)
        self.rnn_z_4 = nn.GRU(128, self.rnn_hidden_sz_list[3], 1, bidirectional=True)
        self.rnn_z_5 = nn.GRU(128, self.rnn_hidden_sz_list[4], 1, bidirectional=True)
        self.rnn_z_6 = nn.GRU(256, self.rnn_hidden_sz_list[5], 1, bidirectional=True)

        self.conv_6 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        self.bn_6 = nn.BatchNorm2d(512)

        self.conv_7 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        self.bn_7 = nn.BatchNorm2d(256)

        self.dp = nn.Dropout(p=0.3)

        self.conv_8 = nn.Conv2d(256, cfg.num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        hidden_list = self.init_hidden(x.size(0))
        x = x.unsqueeze(-1).permute(0, 3, 2, 1)
        
        x_np = x.cpu().numpy()
        x_slice_idx = torch.from_numpy(gen_slice_idx(x_np, self.resolution[0], 0).astype('int32')).cuda()
        y_slice_idx = torch.from_numpy(gen_slice_idx(x_np, self.resolution[1], 1).astype('int32')).cuda()
        z_slice_idx = torch.from_numpy(gen_slice_idx(x_np, self.resolution[2], 2).astype('int32')).cuda()

        num_batch, _, num_points, _ = x.size()

        x_hidden_1, x_hidden_2, x_hidden_3, x_hidden_4, x_hidden_5, x_hidden_6, y_hidden_1, y_hidden_2, y_hidden_3, y_hidden_4, y_hidden_5, y_hidden_6, z_hidden_1, z_hidden_2, z_hidden_3, z_hidden_4, z_hidden_5, z_hidden_6 = hidden_list

        # -- conv block 1
        conv_1 = self.relu(self.bn_1(self.conv_1(x)))  # num_batch, 64, num_points, 1
        conv_2 = self.relu(self.bn_2(self.conv_2(conv_1)))  # num_batch, 64, num_points, 1
        conv_3 = self.relu(self.bn_3(self.conv_3(conv_2)))  # num_batch, 64, num_points, 1

        def pool(x, idx, num_slices):
            res = torch.zeros((num_batch, x.size(1), num_slices, 1)).cuda()
            for i in range(num_slices):
                mask = (idx == i).unsqueeze(1).unsqueeze(-1).expand(num_batch, x.size(1), num_points, 1).float()
                res[:, :, i, :] = torch.max(x * mask, 2)[0]
            return res
        # -- RNN block
        x_pooled = pool(conv_3, x_slice_idx, self.num_slice[0])  # num_batch, 64, numSlices, 1
        y_pooled = pool(conv_3, y_slice_idx, self.num_slice[1])
        z_pooled = pool(conv_3, z_slice_idx, self.num_slice[2])

        x_pooled = x_pooled[:, :, :, 0].permute(2, 0, 1).contiguous()
        y_pooled = y_pooled[:, :, :, 0].permute(2, 0, 1).contiguous()
        z_pooled = z_pooled[:, :, :, 0].permute(2, 0, 1).contiguous()

        x_rnn_1, _ = self.rnn_x_1(x_pooled, x_hidden_1)
        x_rnn_2, _ = self.rnn_x_2(x_rnn_1, x_hidden_2)
        x_rnn_3, _ = self.rnn_x_3(x_rnn_2, x_hidden_3)
        x_rnn_4, _ = self.rnn_x_4(x_rnn_3, x_hidden_4)
        x_rnn_5, _ = self.rnn_x_5(x_rnn_4, x_hidden_5)
        x_rnn_6, _ = self.rnn_x_6(x_rnn_5, x_hidden_6)

        y_rnn_1, _ = self.rnn_y_1(y_pooled, y_hidden_1)
        y_rnn_2, _ = self.rnn_y_2(y_rnn_1, y_hidden_2)
        y_rnn_3, _ = self.rnn_y_3(y_rnn_2, y_hidden_3)
        y_rnn_4, _ = self.rnn_y_4(y_rnn_3, y_hidden_4)
        y_rnn_5, _ = self.rnn_y_5(y_rnn_4, y_hidden_5)
        y_rnn_6, _ = self.rnn_y_6(y_rnn_5, y_hidden_6)

        z_rnn_1, _ = self.rnn_z_1(z_pooled, z_hidden_1)
        z_rnn_2, _ = self.rnn_z_2(z_rnn_1, z_hidden_2)
        z_rnn_3, _ = self.rnn_z_3(z_rnn_2, z_hidden_3)
        z_rnn_4, _ = self.rnn_z_4(z_rnn_3, z_hidden_4)
        z_rnn_5, _ = self.rnn_z_5(z_rnn_4, z_hidden_5)
        z_rnn_6, _ = self.rnn_z_6(z_rnn_5, z_hidden_6)

        # -- uppooling
        x_rnn_6 = x_rnn_6.permute(1, 2, 0).contiguous()
        x_rnn_6 = x_rnn_6.view(x_rnn_6.size(0), x_rnn_6.size(1), x_rnn_6.size(2), 1)

        y_rnn_6 = y_rnn_6.permute(1, 2, 0).contiguous()
        y_rnn_6 = y_rnn_6.view(y_rnn_6.size(0), y_rnn_6.size(1), y_rnn_6.size(2), 1)

        z_rnn_6 = z_rnn_6.permute(1, 2, 0).contiguous()
        z_rnn_6 = z_rnn_6.view(z_rnn_6.size(0), z_rnn_6.size(1), z_rnn_6.size(2), 1)

        def unpool(x, slice_idx):
            return torch.gather(x, 2, slice_idx.unsqueeze(-1).unsqueeze(1).expand(-1, x.size(1), -1, -1).long())
        x_rnn_6 = unpool(x_rnn_6, x_slice_idx)
        y_rnn_6 = unpool(y_rnn_6, y_slice_idx)
        z_rnn_6 = unpool(z_rnn_6, z_slice_idx)

        # -- conv block 3
        rnn = x_rnn_6 + y_rnn_6 + z_rnn_6

        conv_6 = self.relu(self.bn_6(self.conv_6(rnn)))  # num_batch, 512, num_points, 1
        conv_7 = self.relu(self.bn_7(self.conv_7(conv_6)))  # num_batch, 256, num_points, 1
        # droped = self.dp(conv_7)
        conv_8 = self.conv_8(conv_7)
        embedding = conv_8.squeeze(3).permute(0, 2, 1)
        return embedding

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def init_hidden(self, bs):
        hidden_list = []
        for i in range(3):
            for hid_sz in self.rnn_hidden_sz_list:
                if self.rnn_type == 'LSTM':
                    hidden_list.append((torch.zeros(2, bs, hid_sz).cuda(),
                                        torch.zeros(2, bs, hid_sz).cuda()))
                else:
                    hidden_list.append(torch.zeros(2, bs, hid_sz).cuda())
        return hidden_list

SaliencyModel = RSNet
CorrespondenceModel = RSNet