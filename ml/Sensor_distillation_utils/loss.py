
#IMPORTS

import torch #for making, training the model and processing the data in pytorch

import torch.nn as nn #neural network construction


import torch.nn.functional as F #used for layer construction in model, in this case used under encoder class



class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        mse = nn.MSELoss()(pred, target)
        rmse = torch.sqrt(mse)
        return rmse
    

class correlation_coefficient_loss_joint_pytorch(nn.Module):

  def __init__(self):
    super(correlation_coefficient_loss_joint_pytorch, self).__init__()

  def forward(self, y_true, y_pred):

    # Calculate mean values
    mx = torch.mean(y_true)
    my = torch.mean(y_pred)

    # Calculate differences from mean
    xm, ym = y_true - mx, y_pred - my

    # Calculate numerator and denominator of Pearson correlation coefficient
    r_num = torch.sum(torch.mul(xm, ym))
    r_den = torch.sqrt(torch.mul(torch.sum(torch.square(xm)), torch.sum(torch.square(ym))))

    # Calculate Pearson correlation coefficient
    r = r_num / r_den

    # Clamp r between -1 and 1
    r = torch.clamp(r, -1.0, 1.0)

    # Calculate l2 loss
    l2 = 1 - torch.square(r)

    # Calculate l1 loss
    l1 = torch.sqrt(F.mse_loss(y_pred, y_true))

    return l1 + l2