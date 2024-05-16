import torch
import torchvision
from pycox.models.loss import CoxPHLoss
from data.constants import NUM_CLASSES

def CoxPH(log_h, events, duration):

    loss = CoxPHLoss()
    return loss(log_h, events, duration)

def focal_binary_cross_entropy(logits, targets, gamma=2):
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = l
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = NUM_CLASSES*loss.mean()
    return loss

def MultilabelBCELoss(logits, targets, reduction='mean'):
    p = targets
    q = logits

    qterm1 = torch.log(q)
    qterm2 = torch.log(torch.ones_like(q) - q)

    batch_loss = -1 * (p*qterm1 + (torch.ones_like(p) - p) * qterm2)
    
    if reduction == 'mean':
        return torch.mean(batch_loss)
    if reduction == 'sum':
        return torch.sum(batch_loss)


