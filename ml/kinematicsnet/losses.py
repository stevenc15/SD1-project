import torch
import torch.nn as nn
import torch.nn.functional as F

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
        mx = torch.mean(y_true)
        my = torch.mean(y_pred)

        xm, ym = y_true - mx, y_pred - my

        r_num = torch.sum(torch.mul(xm, ym))
        r_den = torch.sqrt(torch.mul(torch.sum(torch.square(xm)), torch.sum(torch.square(ym))))

        r = r_num / r_den
        r = torch.clamp(r, -1.0, 1.0)

        l2 = 1 - torch.square(r)
        l1 = torch.sqrt(F.mse_loss(y_pred, y_true))

        return l1 + l2
