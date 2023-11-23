import torch
import torch.nn.functional as F

def focal_loss(preds, targets, alpha = .25, gamma = 2):
    CE_loss = F.cross_entropy(preds, targets, reduction='none')
    pt = torch.exp(-CE_loss) # prevents nans when probability 0
    F_loss = alpha * (1-pt)**gamma * CE_loss
    return F_loss.mean()