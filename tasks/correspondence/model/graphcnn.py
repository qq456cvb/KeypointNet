import warnings
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy
from sklearn import neighbors
from torch.autograd import Variable


class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx


def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    # Weights.
    sigma2 = np.mean(dist[:, -1]) ** 2
    # print sigma2
    dist = np.exp(- dist ** 2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W


def normalize_adj(adj):
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalized_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    norm_laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    return norm_laplacian


def scaled_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian


def pc2lap(pcd, knn=20):
    graph = neighbors.kneighbors_graph(pcd, knn, mode='distance', include_self=False)
    graph = graph.toarray()
    conns = np.sum(graph > 0, axis=-1)
    graph = np.exp(-graph ** 2 / (np.sum(graph, axis=-1, keepdims=True) / conns[:, None]) ** 2) * (graph > 0).astype(np.float32)
    return scaled_laplacian(graph)


class GraphConvNet(nn.Module):

    def __init__(self, net_parameters, mlps):

        print('Graph ConvNet: LeNet5')

        super().__init__()

        # parameters
        D, CL1_F, CL1_K, CL2_F, CL2_K = net_parameters
        feature1, feature2 = mlps

        # graph CL1
        self.cl1 = nn.Linear(D * CL1_K, CL1_F)
        Fin = CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K
        self.CL1_F = CL1_F

        # graph CL2
        self.cl2 = nn.Linear(CL2_K * CL1_F, CL2_F)
        Fin = CL2_K * CL1_F
        Fout = CL2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.cl2.weight.data.uniform_(-scale, scale)
        self.cl2.bias.data.fill_(0.0)
        self.CL2_K = CL2_K
        self.CL2_F = CL2_F

        self.fc1 = torch.nn.Linear(CL2_F, feature1)
        self.fc2 = torch.nn.Linear(feature1, feature2)

        # nb of parameters
        nb_param = CL1_K * CL1_F + CL1_F  # CL1
        nb_param += CL2_K * CL1_F * CL2_F + CL2_F  # CL2
        print('nb of parameters=', nb_param, '\n')

    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt(2.0 / (Fin + Fout))
        W.uniform_(-scale, scale)

        return W

    def graph_conv_cheby(self, x, cl, L, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size()
        B, V, Fin = int(B), int(V), int(Fin)

        # convert scipy sparse matric L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data)
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = Variable(L, requires_grad=False)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin * B])  # V x Fin*B
        x = x0.unsqueeze(0)  # 1 x V x Fin*B

        def concat(x, x_):
            x_ = x_.unsqueeze(0)  # 1 x V x Fin*B
            return torch.cat((x, x_), 0)  # K x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm()(L, x0)  # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])  # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B * V, Fin * K])  # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)  # B*V x Fout
        x = x.view([B, V, Fout])  # B x V x Fout

        return x

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)  # B x F x V/p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x

    def forward(self, x):
        x = x.permute(0, 2, 1)
        embeddings = []
        for b in range(x.size(0)):
            x_np = x[b].cpu().numpy()

            # TODO: precompute Laplacian
            L = pc2lap(x_np)
            # graph CL1
            # x = x.unsqueeze(2)  # B x V x Fin=1
            y = self.graph_conv_cheby(x[b].unsqueeze(0), self.cl1, L, self.CL1_F, self.CL1_K)
            y = F.relu(y)

            # graph CL2
            y = self.graph_conv_cheby(y, self.cl2, L,  self.CL2_F, self.CL2_K)
            y = F.relu(y)
            embeddings.append(y)
        embeddings = torch.cat(embeddings)

        embeddings = F.relu(self.fc1(embeddings))
        embeddings = F.relu(self.fc2(embeddings))

        # # FC1
        # x = x.view(-1, self.FC1Fin)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = nn.Dropout(d)(x)
        #
        # # FC2
        # x = self.fc2(x)

        return embeddings #, embeddings
