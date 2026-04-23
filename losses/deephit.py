import torch
import torchvision
from pycox.models.loss import CoxPHLoss
# def pad_col(input, val=0, where='end'):
#     """Addes a column of `val` at the start of end of `input`."""
#     if len(input.shape) != 2:
#         raise ValueError(f"Only works for `phi` tensor that is 2-D.")
#     pad = torch.zeros_like(input[:, :1])
#     if val != 0:
#         pad = pad + val
#     if where == 'end':
#         return torch.cat([input, pad], dim=1)
#     elif where == 'start':
#         return torch.cat([pad, input], dim=1)
#     raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

# def DeephitCRLoss(phi, events, durations, sigma):
#     '''
#     Args: 
#         phi - tensor with shape [BatchSize, num_labels, num_durations]
#         events - tensor with shape [BatchSize, num_labels, num]
#         durations - tensor with shape [BatchSize, num_labels, num_durations]
#     '''
#     idx_durations = idx_durations.view(-1)
#     events = events.view(-1) - 1
#     event_01 = (events == -1).float()

#     batch_size, n_risks = phi.shape[:2]
#     pmf = pad_col(phi.view(batch_size, -1)).softmax(1)
#     pmf = pmf[:, :-1].view(phi.shape)
#     y = torch.zeros_like(pmf)
#     y[torch.arange(batch_size), :, idx_durations] = 1.

def CoxPH(log_h, events, duration):
    return CoxPHLoss(log_h, events, duration)
