import torch as t
import numpy as np
from params import args
import torch.nn.functional as F

def calc_reg_loss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret

def contrast(nodes, allEmbeds, allEmbeds2=None):
    if allEmbeds2 is not None:
        pckEmbeds = allEmbeds[nodes]
        scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
    else:
        uniqNodes = t.unique(nodes)
        pckEmbeds = allEmbeds[uniqNodes]
        scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
    return scores

def calc_reward(lastLosses, eps):
    if len(lastLosses) < 3:
        return 1.0
    curDecrease = lastLosses[-2] - lastLosses[-1]
    avgDecrease = 0
    for i in range(len(lastLosses) - 2):
        avgDecrease += lastLosses[i] - lastLosses[i + 1]
    avgDecrease /= len(lastLosses) - 2
    return 1 if curDecrease > avgDecrease else eps

def calc_sigmoid_reward(lastLosses, eps):
    if len(lastLosses) < 3:
        return 1.0
    curDecrease = lastLosses[-2] - lastLosses[-1]
    avgDecrease = 0
    for i in range(len(lastLosses) - 2):
        avgDecrease += lastLosses[i] - lastLosses[i + 1]
    avgDecrease /= len(lastLosses) - 2
    return max(t.sigmoid(curDecrease.detach().cpu() / avgDecrease.detach().cpu()), eps)

def calc_min_reward(lastLosses):
    if len(lastLosses) < 3:
        return 1.0
    curDecrease = lastLosses[-2] - lastLosses[-1]
    avgDecrease = 0
    for i in range(len(lastLosses) - 2):
        avgDecrease += lastLosses[i] - lastLosses[i + 1]
    avgDecrease /= len(lastLosses) - 2
    return min(curDecrease.detach().cpu().numpy() / avgDecrease.detach().cpu().numpy(), 1)

def cross_entropy(seq_out, pos_emb, neg_emb, tar_msk):
    seq_emb = seq_out.view(-1, args.latdim)
    pos_emb = pos_emb.view(-1, args.latdim)
    neg_emb = neg_emb.view(-1, args.latdim)
    pos_scr = t.sum(pos_emb * seq_emb, -1)
    neg_scr = t.sum(neg_emb * seq_emb, -1)
    tar_msk = tar_msk.view(-1).float()
    loss = t.sum(
        - t.log(t.sigmoid(pos_scr) + 1e-24) * tar_msk -
        t.log(1 - t.sigmoid(neg_scr) + 1e-24) * tar_msk
    ) / t.sum(tar_msk)
    return loss
