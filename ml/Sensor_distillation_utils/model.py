
#IMPORTS

import torch #for making, training the model and processing the data in pytorch

import torch.nn as nn #neural network construction


import torch.nn.functional as F #used for layer construction in model, in this case used under encoder class


class Encoder_GRU(nn.Module):
    def __init__(self, input_shape_1D, input_shape_2D, dropout, w):
        super(Encoder_GRU, self).__init__()
        self.lstm_1 = nn.GRU(input_shape_1D, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)
        self.output_GRU = nn.Linear(w*128, 6 * w)

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)


    def forward(self, x, w):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)
        out_2=self.flatten(out_2)

        output_GRU=self.output_GRU(out_2).view(-1,w,6)

        return output_GRU

class Encoder_GRU_cnn(nn.Module):
    def __init__(self, input_dim, dropout, w):
        super(Encoder_GRU_cnn, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)
        self.output_GRU = nn.Linear(w*128, 6 * w)



    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)
        out_2=self.flatten(out_2)

        return out_2

class Encoder_CNN_1D(nn.Module):
    def __init__(self, w, input_shape_1D, input_shape_2D, dropout, hidden_dim=64, output_size=128, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_shape_1D, hidden_dim, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(hidden_dim, output_size, kernel_size, stride, padding)
        self.conv4 = nn.Conv1d(output_size, output_size, kernel_size, stride, padding)
        self.BN_2= nn.BatchNorm1d(hidden_dim)
        self.BN_4= nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten=nn.Flatten()

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)
        self.BN_2D= nn.BatchNorm2d(input_shape_2D, affine=False)

        self.model_2=Encoder_GRU_cnn(input_shape_1D,0.40)

        self.output_C2 = nn.Linear(25*32+w*128, 6 * w)

    def forward(self, x, x_1, w):

        input_1D_N_1=x.transpose(1,2)
        input_1D_N_1=self.BN_1D(input_1D_N_1)
        input_1D_N=input_1D_N_1.transpose(1,2)

        # input_1D_N_1=x_1.transpose(1,2)
        # input_1D_N_1=self.BN_1D(input_1D_N_1)
        # input_1D_N=input_1D_N_1.transpose(1,2)



        x = input_1D_N.transpose(1, 2)  # reshape from (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.BN_2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.BN_4(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # reshape back to (batch_size, seq_len, output_size)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.flatten(x)

        model_2_output = self.model_2(input_1D_N)
        model_2_output_1 = torch.cat([model_2_output, x], dim=-1)

        output_C2=self.output_C2(model_2_output_1).view(-1,w,6)

        return output_C2
    
class Encoder_CNN_2D(nn.Module):
    def __init__(self, w, input_shape_1D,input_shape_2D, dropout, hidden_dim=64, output_size=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape_2D, hidden_dim, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(hidden_dim, output_size, kernel_size, stride, padding)
        self.BN_2= nn.BatchNorm2d(hidden_dim)
        self.BN_4= nn.BatchNorm2d(output_size)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten=nn.Flatten()

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)
        self.BN_2D= nn.BatchNorm2d(input_shape_2D, affine=False)

        self.model_3=Encoder_GRU_cnn(input_shape_1D,0.40)

        self.output_C2 = nn.Linear(25*32+w*128, 6 * w)

    def forward(self, x_1,x, w):

        input_1D_N_1=x_1.transpose(1,2)
        input_1D_N_1=self.BN_1D(input_1D_N_1)
        input_1D_N=input_1D_N_1.transpose(1,2)

        inputs_2D_N=x.transpose(1,3)
        inputs_2D_N=self.BN_2D(inputs_2D_N)
        inputs_2D_N=inputs_2D_N.transpose(1,3)

        x = inputs_2D_N.transpose(1, 3)  # reshape from (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
        x = F.relu(self.conv1(x))
        x = self.BN_2(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.BN_4(x)
        x = self.pool(x)
        x = x.transpose(1, 3)  # reshape back to (batch_size, seq_len, output_size)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.flatten(x)

        model_3_output = self.model_3(input_1D_N)
        model_3_output = torch.cat([model_3_output, x], dim=-1)
        # output_C2 = self.output_C2(model_3_output)

        output_C1=self.output_C2(model_3_output).view(-1,w,6)

        return output_C1
    
