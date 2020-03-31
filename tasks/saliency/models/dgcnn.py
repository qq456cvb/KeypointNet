import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * t.matmul(x.transpose(2, 1), x)
    xx = t.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = t.device('cuda')

    idx_base = t.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = t.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class DGCNN(nn.Module):
    def __init__(self, output_channels=10, cfg=None):
        super(DGCNN, self).__init__()
        cfg = {"emb_dims": 128,
               "drop_prob1": 0.1,
               "drop_prob2": 0.45}
        cfg["emb_dims"] = 128
        self.cfg = cfg
        self.k = 20
        # self.stn = STNkd(3);
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.cfg['emb_dims'])

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),  # *2����Ϊÿ�ζ���cat����������
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.cfg['emb_dims'], kernel_size=1, bias=False),  # ���һ����conv1d����Ϊû��k��
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.cfg['emb_dims'] * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.cfg['drop_prob1'])
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.cfg['drop_prob2'])
        self.linear3 = nn.Conv1d(768, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        # trans = self.stn(x)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, trans)
        # x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.k)  # 3 * 2048 -> 6 * 2048 * k
        x = self.conv1(x)  # 6 * 2048 * 20 -> 64 * 2048 * 20
        x1 = x.max(dim=-1, keepdim=False)[0]  # 64 * 2048 * 20 -> 64 * 2048
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        point_feat = t.cat((x1, x2, x3, x4), dim=1)
        # embedding = point_feat.clone()

        x = self.conv5(point_feat)  # (64 + 64 + 128 + 256) * 2048-> emb * 2048
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # emb * 2048 -> emb
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # emb * 2048 -> emb
        x = t.cat((x1, x2), 1)
      
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # 2 * emb -> 512
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # 512 -> 256
        x = self.dp2(x)

        x = x.view(batch_size, 256, 1).repeat(1, 1, point_feat.size(-1))  # B * 256 -> B * 256 * 2048
        x = t.cat((point_feat, x), dim=1)  # B * 256 * 2048, B * 512 * 2048 -> B * 768 * 2048
        embedding = x.clone()
        x = self.linear3(x)  # B * 768 * 2048 -> B * 10 * 2048

        return x.transpose(1, 2).contiguous()#, embedding.transpose(1, 2).contiguous()



if __name__ == '__main__':
    anchors = t.tensor(
        [
            [[1, 2, 3]],
            [[4, 5, 6]]
        ]
    ).float()
    pts = t.tensor(
        [
            [[1.1, 2.2, 3.3]],
            [[4.4, 5.5, 6.6]]
        ]
    )
    print(pts.device)

