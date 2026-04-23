import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class GradientBlender:

    '''
    Implements gradient blending outlined in What Makes Training Multi-modal Classification Networks Hard? https://arxiv.org/pdf/1905.12681.pdf , Facebook AI

    Usage:
        This class forms a wrapper around your loss function

    '''
    def __init__(self, loss_function, survival=False, reduction='sum', device='cpu', surv_criterion=None):
        self.loss_function = loss_function
        self.weights = None # Need some form of initial weights
        self.reduction = reduction.lower()
        self.survival = survival
        self.lvn = None
        self.ltn = None
        self.device=device
        self.surv_criterion=surv_criterion
        self.history = []
        self.train_modality_history = []
        self.test_modality_history = []

    def updateWeightsClass(self, train_preds, train_targs, val_preds, val_targs):

        train_preds = train_preds.to(self.device)
        train_targs = train_targs.to(self.device)
        val_preds = val_preds.to(self.device)
        val_targs = val_targs.to(self.device)

        train_loss = self.computeLoss(train_preds, train_targs, reduceToHeads = True)
        val_loss = self.computeLoss(val_preds, val_targs, reduceToHeads = True)


        # For the first iteration, equally weight all heads and set lvn and ltn
        if self.lvn is None or self.ltn is None:
            self.weights = self.normalize(torch.ones(train_preds.shape[0])).to(self.device)

        # Otherwise, perform gradient blending method outlined in referenced paper
        else:
            o_n = self.lvn - self.ltn
            o_npn = val_loss - train_loss
            delta_g = val_loss - self.lvn
            delta_o = o_npn - o_n

            gbEstimate = delta_g / torch.pow(delta_o, 2)

            self.weights = self.normalize(gbEstimate).to(self.device)

        self.lvn = val_loss
        self.ltn = train_loss

    def updateWeightsSurv(self, train_preds, train_events, train_durations, val_preds, val_events, val_durations):
        train_loss = self.computeLossSurv(train_preds, train_events, train_durations, reduceToHeads = True)
        val_loss = self.computeLossSurv(val_preds, val_events, val_durations, reduceToHeads = True)

        # For the first iteration, equally weight all heads and set lvn and ltn
        if self.lvn is None or self.ltn is None:
            self.weights = self.normalize(torch.ones(train_preds.shape[0])).to(self.device)

        # Otherwise, perform gradient blending method outlined in referenced paper
        else:
            o_n = self.lvn - self.ltn   # Compute overfitting at checkpoint N
            o_npn = val_loss - train_loss  # Compute overfitting at checkpoint N + n
            delta_g = val_loss - self.lvn # Compute change in generalization between the two checkpoints
            delta_o = o_npn - o_n  # Compute change in overfitting between the two checkpoints

            gbEstimate = delta_g / torch.pow(delta_o, 2) # Find G/O^2 ratios

            self.weights = self.normalize(gbEstimate).to(self.device) # Normalize G/O^2 ratios

        # Losses from current checkpoint (N + n) becomes the previous checkpoint losses (N) for the next iteration
        self.lvn = val_loss
        self.ltn = train_loss


        self.history.append(self.weights.detach().cpu().numpy())
        self.train_modality_history.append(train_loss.detach().cpu().numpy())
        self.test_modality_history.append(val_loss.detach().cpu().numpy())

    def updateWeights(self, *args, **kwargs):
        if self.survival:
            self.updateWeightsSurv(*args, **kwargs)
        else:
             self.updateWeightsClass(*args, **kwargs)

    def computeLoss(self, *args, **kwargs):
         if self.survival:
              return self.computeLossSurv(*args, **kwargs)
         else:
              return self.computeLossClassification(*args, **kwargs)
         
    def computeLossClassification(self, preds, targets, reduceToHeads=False, no_reduce=False):
        '''
        args:
            preds - k+1 x N x C tensor
                N x C prediction tensors for each modality head (+1 for multimodality output head), stacked along 0 dimension
                N is the batch size, C is the number of classes
            targets - N x C tensor
                N x C Ground tensor containing ground truth values
            epoch - current epoch number
        
        returns:
            k+1 x N x C loss tensor
        '''

        # Stack the target vector to match dimensionality of prediction tensor
        targets = torch.stack([targets for _ in range(preds.shape[0])], dim=0)

        loss = self.loss_function(preds, targets)
        if self.weights is None:
             self.weights = self.normalize(torch.ones(preds.shape[0]))
             self.history.append(self.weights.detach().cpu().numpy())

        if no_reduce:
             return loss
        loss = self.reduceToHeads(loss)
        if reduceToHeads:
             return loss
        
        self.weights = self.weights.to(device=self.device)
        return self.reduce(self.weights * loss)
             
    def computeLossSurv(self, preds, events, durations, reduceToHeads=False):
        '''
        args:
            preds - k+1 x N x C tensor
                N x C prediction tensors for each modality head (+1 for multimodality output head), stacked along 0 dimension
                N is the batch size, C is the number of classes
            events - N x C tensor
                N x C tensor containing ground truth values of whether or not event is censored
            durations -  N x C tensor
                N x C tensor containing ground truth values of the maximum duration prior to an event occuring or being censored
            epoch - current epoch number
        
        returns:
            k+1 x N x C loss tensor
        '''

        head_losses = torch.stack([self.surv_criterion(self.loss_function, preds[i,...], events, durations,self.device) for i in range(preds.shape[0])], dim=0)

        if self.weights is None:
             self.weights = self.normalize(torch.ones(preds.shape[0]))

        if reduceToHeads:
             return head_losses
        
        return self.reduce(self.weights * head_losses), head_losses[0]

    def reduceToHeads(self, loss):
         '''
         reduces loss along every dimension except the first
         The first dimension contains one tensor for each output head, so this will return a k+1 x 1 size tensor containing reduced loss for each modality
         (+1 for the multimodal head)

         args: 
            loss - unreduced k+1 x N x C unreduced loss tensor
        returns
            k+1 x 1 loss tensor containing reduced loss values for each modality (+1 for multimodal head)
            if reduction is set to None, this function acts as a no-op 
         '''
         if self.reduction.startswith('sum'):
              return torch.sum(loss, dim=(1,2))
         elif self.reduction.startswith('mean'):
              return torch.mean(loss, dim=(1,2))
         elif self.reduction.startswith('none') or self.reduction is None:
              return loss
         else:
              raise ValueError('Unable to reduce loss, unrecognized reduction: {}'.format(self.reduction))
         
    def reduce(self, loss):
         '''
         reduces loss along every dimension

         args: 
            loss - unreduced loss tensor with arbitrary shape
        returns
            loss tensor containing fully reduced loss values
            if reduction is set to None, this function acts as a no-op 
         '''
         if self.reduction.startswith('sum'):
              return torch.sum(loss)
         elif self.reduction.startswith('mean'):
              return torch.mean(loss)
         elif self.reduction.startswith('none') or self.reduction is None:
              return loss
         else:
              raise ValueError('Unable to reduce loss, unrecognized reduction: {}'.format(self.reduction))
         
    def normalize(self, weights):
         '''
         Normalizes input tensor to sum to 1

         args: weights - 1D tensor containing k+1 elements corresponding to weights of each output head
         '''
         return F.softmax(weights)
    
    def saveHistory(self):
         history = np.array(self.history)
         train_losses = np.array(self.train_modality_history)
         val_losses = np.array(self.test_modality_history)
         np.savetxt('gblend_weights_history.csv', history, delimiter=',')
         np.savetxt('train_modality_loss_history.csv', train_losses, delimiter=',')
         np.savetxt('test_modality_loss_history.csv', val_losses, delimiter=',')