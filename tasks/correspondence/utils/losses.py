import torch
import torch.nn as nn
import torch.nn.functional as F

class PCKLoss(nn.Module):
    """
    Calculate the cross entropy between pred logits and one hot kp labels.
    """
    def __init__(self):
        super(PCKLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred, kps):
        pred = self.softmax(pred)
        loss = F.binary_cross_entropy(pred, kps)
        pred_kp_idx = torch.argmax(pred, dim=1)
        return loss, pred_kp_idx

class BinaryLoss(nn.Module):
    """
    Calculate the cross entropy between pred logits and one hot kp labels.
    """
    def __init__(self):
        super(BinaryLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pred, kps):
        pred = self.softmax(pred)
        loss = F.binary_cross_entropy(pred, kps)
        pred_kp = torch.argmax(pred, dim=-1)
        return loss, pred_kp






