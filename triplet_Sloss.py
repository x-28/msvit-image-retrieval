import math

from torch import nn
import torch
from torch.nn import functional as F
from config import opt

import numpy as np
def EuclideanDistances(a,b):
    sq_a = a.repeat(8,1)
    sq_b = b.repeat(1,8)
    sq_b = sq_b.reshape(64,64)
    sq_b = sq_b
    x = (sq_a - sq_b) ** 2
    sum_x = torch.sum(x,dim=1).unsqueeze(0)  # m->[m, 1]
    sqrt_sum_x = torch.sqrt(sum_x)
    Eucdistance = sqrt_sum_x.reshape(8,8)

    return Eucdistance



def cos_distance1(source, target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)
    a_,b_ = source, target
    yuandian = torch.zeros([8,64],device = device)
    a = EuclideanDistances(a_,yuandian)
    b = EuclideanDistances(b_,yuandian)

    # #计算余弦值
    # #矩阵相乘
    # c = torch.mm(a_,b_.t())
    # #模长
    # d = torch.mul(a,b)
    # cos = c/d
    #计算任意三角形面积 半周长 p=0.5*(a+b+c) ，面积S = p*(p-a)*(p-b)*(p-c))
    # b1 = b.repeat(8,1)
    # aa = a.repeat(8,1)
    # bb = b1.t()
    c = EuclideanDistances(source, target)
    #cc =torch.where(torch.isnan(c), torch.full_like(c, 0), c)
    p1 = a+b+c
    p = 0.5 * p1
    s = (p-a) *(p-b) * (p-c) * p
    ma2 = torch.max(s)
    mi2 = torch.min(s)
    s = torch.clamp(s, 0)
    S = s ** 0.5


    #distances = torch.clamp(1 - cos_sim, 0)
    # t = (S-mi) / (ma-mi)

    distances = torch.clamp(S, 0)

    return distances
    #return c


def get_triplet_mask(s_labels, t_labels, opt):
    batch_size = s_labels.shape[0]
    sim_origin = s_labels.mm(t_labels.t())
    sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., batch_size) + 2
    ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
    th = torch.log2(ph).to(opt.device)
    Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
    sim_origin = 2 ** sim_origin - 1
    sim_origin = sim_origin / Z
    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)
    sim_pos = sim_origin.unsqueeze(2)
    sim_neg = sim_origin.unsqueeze(1)
    weight = sim_pos - sim_neg
    mask = i_equal_j * (1 - i_equal_k)
    return mask, weight
    
class TripletLoss(nn.Module):
    def __init__(self, opt, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.reduction = reduction
        self.opt = opt

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        pairwise_dist1 = cos_distance1(source, target)
#       pairwise_dist2 = cos_distance(source, target)
        pairwise_dist = pairwise_dist1
        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
         # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask, weight = get_triplet_mask(s_labels, t_labels, self.opt)
        #triplet_loss1 = weight * mask * triplet_loss
        triplet_loss1 = weight * mask * triplet_loss
        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = triplet_loss1.clamp(0)
        ma4 = torch.max(triplet_loss)
        mi4 = torch.min(triplet_loss)
        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss


