import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_1(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_1, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 16, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(32, 8, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2

class Encoder_2(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_2, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 16, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(32, 8, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2

class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class Kinematics_lightweight(nn.Module):
    def __init__(self, input_acc, input_gyr,drop_prob=0.05):
        super(Kinematics_lightweight, self).__init__()

        self.encoder_1_acc=Encoder_1(input_acc, drop_prob)
        self.encoder_1_gyr=Encoder_1(input_gyr, drop_prob)

        self.encoder_2_acc=Encoder_2(input_acc, drop_prob)
        self.encoder_2_gyr=Encoder_2(input_gyr, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)

        self.fc = nn.Linear(2*2*16+16,1)

        self.dropout=nn.Dropout(p=0.05)

        self.gate_1=GatingModule(16)
        self.gate_2=GatingModule(16)
        self.gate_3=GatingModule(16)

        self.fc_kd = nn.Linear(2*16, 16)

        # Define the gating network
        self.weighted_feat = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid())

        self.attention=nn.MultiheadAttention(2*16,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(16*2, 2*16), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*2*16+16, 2*2*16+16), nn.Sigmoid())


    def forward(self, x_acc, x_gyr,w=100):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)

        x_acc_2=x_acc_1.view(-1, w, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, w, x_gyr_1.size(-1))

        x_acc_1=self.encoder_1_acc(x_acc_2)
        x_gyr_1=self.encoder_1_gyr(x_gyr_2)

        x_acc_2=self.encoder_2_acc(x_acc_2)
        x_gyr_2=self.encoder_2_gyr(x_gyr_2)

        x_acc=self.gate_1(x_acc_1,x_acc_2)
        x_gyr=self.gate_2(x_gyr_1,x_gyr_2)

        x=torch.cat((x_acc,x_gyr),dim=-1)

        x_kd=self.fc_kd(x)
        out_1, attn_output_weights=self.attention(x,x,x)
        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        weights_1 = self.weighted_feat(x[:,:,0:16])
        weights_2 = self.weighted_feat(x[:,:,16:2*16])

        x_1=weights_1*x[:,:,0:16]
        x_2=weights_2*x[:,:,16:2*16]

        out_3=x_1+x_2

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out

