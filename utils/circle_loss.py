import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from torch.autograd import Variable


def similarity(batch_feature, features, batch_label, labels, bit):
    similarity_matrix = batch_feature @ features.transpose(1, 0)
    similarity_matrix = similarity_matrix / bit
    label_matrix = (batch_label.mm(labels.t()) > 0)
    
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1) 
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        # ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        # an = torch.clamp_min(sn.detach() + self.m, min=0.)

        # delta_p = 1 - self.m
        # delta_n = self.m

        # logit_p = - ap * (sp - delta_p) * self.gamma
        # logit_n = an * (sn - delta_n) * self.gamma

        ap = torch.clamp_min(- sp.detach() + 2, min=0.)
        an = torch.clamp_min(sn.detach() + 2, min=0.)

        logit_p = - ap * sp * self.gamma
        logit_n = an * sn * self.gamma
        # logit_p = - sp * self.gamma
        # logit_n = sn * self.gamma

        # loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        loss = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)

        return loss
