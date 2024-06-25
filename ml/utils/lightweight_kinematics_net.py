"""# Pytorch Implementation"""

import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy
import statistics
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statistics import stdev
import math
import h5py

import numpy as np
import time

from scipy.signal import butter,filtfilt
import sys
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
import pandas
import matplotlib.pyplot as plt

# from tsf.model import TransformerForecaster


# from tensorflow.keras.utils import np_utils
import itertools
###  Library for attention layers
import pandas as pd
import os
import numpy as np
#from tqdm import tqdm # Processing time measurement
from sklearn.model_selection import train_test_split

import statistics
import gc
import torch.nn.init as init

############################################################################################################################################################################
############################################################################################################################################################################

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.utils.weight_norm as weight_norm
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.parameter import Parameter

from torchsummary import summary



import torch.optim as optim
import gc

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

"""## Data Preparation"""

### IMUs- Chest, Waist, Right Foot, Right shank, Right thigh, Left Foot, Left shank, Left thigh, 2D-body coordinate
### 0:48- IMU, 48:92-2D body coordinate, 92:97-- Target


### Data Processing

batch_size = 64

train_features = torch.Tensor(train_X_1D)
val_features = torch.Tensor(X_validation_1D)
test_features = torch.Tensor(test_X_1D)


train_features_acc_8=torch.cat((train_features[:,:,0:3],train_features[:,:,6:9],train_features[:,:,12:15],train_features[:,:,18:21],train_features[:,:,24:27]\
                             ,train_features[:,:,30:33],train_features[:,:,36:39],train_features[:,:,42:45]),axis=-1)
test_features_acc_8=torch.cat((test_features[:,:,0:3],test_features[:,:,6:9],test_features[:,:,12:15],test_features[:,:,18:21],test_features[:,:,24:27]\
                             ,test_features[:,:,30:33],test_features[:,:,36:39],test_features[:,:,42:45]),axis=-1)
val_features_acc_8=torch.cat((val_features[:,:,0:3],val_features[:,:,6:9],val_features[:,:,12:15],val_features[:,:,18:21],val_features[:,:,24:27]\
                             ,val_features[:,:,30:33],val_features[:,:,36:39],val_features[:,:,42:45]),axis=-1)


train_features_gyr_8=torch.cat((train_features[:,:,3:6],train_features[:,:,9:12],train_features[:,:,15:18],train_features[:,:,21:24],train_features[:,:,27:30]\
                             ,train_features[:,:,33:36],train_features[:,:,39:42],train_features[:,:,45:48]),axis=-1)
test_features_gyr_8=torch.cat((test_features[:,:,3:6],test_features[:,:,9:12],test_features[:,:,15:18],test_features[:,:,21:24],test_features[:,:,27:30]\
                             ,test_features[:,:,33:36],test_features[:,:,39:42],test_features[:,:,45:48]),axis=-1)
val_features_gyr_8=torch.cat((val_features[:,:,3:6],val_features[:,:,9:12],val_features[:,:,15:18],val_features[:,:,21:24],val_features[:,:,27:30]\
                             ,val_features[:,:,33:36],val_features[:,:,39:42],val_features[:,:,45:48]),axis=-1)


train_features_2D = torch.Tensor(train_X_2D)
val_features_2D = torch.Tensor(X_validation_2D)
test_features_2D = torch.Tensor(test_X_2D)

train_targets = torch.Tensor(train_y_5)
val_targets = torch.Tensor(Y_validation)
test_targets = torch.Tensor(test_y)

## all Modality Features
train = TensorDataset(train_features, train_features_2D,train_features_acc_8,train_features_gyr_8, train_targets)
val = TensorDataset(val_features, val_features_2D,val_features_acc_8,val_features_gyr_8, val_targets)
test = TensorDataset(test_features,test_features_2D,test_features_acc_8,test_features_gyr_8,test_targets)


train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)

"""## Important Functions"""

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

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train_kinematics(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()
    # criterion =correlation_coefficient_loss_joint_pytorch()

    # criterion=PearsonCorrLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=learn_rate)
    # optimizer_2 = torch.optim.Adam(model.cnn_1D.parameters(), lr=learn_rate)
    # optimizer_3 = torch.optim.Adam(model.cnn_2D.parameters(), lr=learn_rate)

    # optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data_features_1D, data_features_2D, data_acc,data_gyr, data_targets) in enumerate(train_loader):
            optimizer.zero_grad()

            output_1, output_2, output_3, output= model(data_features_1D[:,:,36:48].to(device).float(),data_features_2D.to(device).float())
                                # Compute the regularization loss for the custom linear layers
            regularization_loss = 0.0
            if hasattr(model.output_GRU, 'regularizer_loss'):
                regularization_loss += model.output_GRU.regularizer_loss()
            if hasattr(model.output_C1, 'regularizer_loss'):
                regularization_loss += model.output_C1.regularizer_loss()
            if hasattr(model.output_C2, 'regularizer_loss'):
                regularization_loss += model.output_C2.regularizer_loss()

            loss=criterion(output_1, data_targets.to(device).float())+criterion(output_2, data_targets.to(device).float())\
            +criterion(output_3, data_targets.to(device).float())+criterion(output, data_targets.to(device).float())+regularization_loss
            loss_1=criterion(output, data_targets.to(device).float())

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_features_1D, data_features_2D,data_acc,data_gyr, data_targets in val_loader:
                output_1, output_2, output_3, output= model(data_features_1D[:,:,36:48].to(device).float(),data_features_2D.to(device).float())
                val_loss += criterion(output, data_targets.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model

"""## Model"""

class RegularizedLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_decay=0.001):
        super(RegularizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, input):
        return self.linear(input)

    def regularizer_loss(self):
        return self.weight_decay * torch.sum(self.linear.bias**2)

class Encoder_GRU(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder_GRU, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)



    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)
        out_2=self.flatten(out_2)

        return out_2

class Encoder_GRU_cnn(nn.Module):
    def __init__(self, input_dim, dropout):
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
    def __init__(self, input_size,dropout, hidden_dim=64, output_size=128, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size, stride, padding)
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

        self.model_2=Encoder_GRU_cnn(12,0.40)

    def forward(self, x, x_1):
        x = x.transpose(1, 2)  # reshape from (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
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

        model_2_output = self.model_2(x_1)
        model_2_output = torch.cat([model_2_output, x], dim=-1)
        # output_C1 = self.output_C1(model_2_output)

        return model_2_output

class Encoder_CNN_2D(nn.Module):
    def __init__(self, input_size,dropout, hidden_dim=64, output_size=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_dim, kernel_size, stride, padding)
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

        self.model_3=Encoder_GRU_cnn(12,0.40)

    def forward(self, x,x_1):
        x = x.transpose(1, 3)  # reshape from (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
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

        model_3_output = self.model_3(x_1)
        model_3_output = torch.cat([model_3_output, x], dim=-1)
        # output_C2 = self.output_C2(model_3_output)

        return model_3_output

class Gait_Net(nn.Module):
    def __init__(self, input_shape_1D,input_shape_2D, w):
        super(Gait_Net, self).__init__()
        self.w = w

        # 1D Models

        self.model_1=Encoder_GRU(12,0.40)
        self.cnn_1D = Encoder_CNN_1D(input_shape_1D,0.40)
        self.cnn_2D = Encoder_CNN_2D(input_shape_2D,0.40)

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)
        self.BN_2D= nn.BatchNorm2d(input_shape_2D, affine=False)

        self.output_GRU = RegularizedLinear(w*128, 6 * w)
        self.output_C1 = RegularizedLinear(25*32+w*128, 6 * w)
        self.output_C2 = RegularizedLinear(25*32+w*128, 6 * w)


    def forward(self, inputs_1D_N, inputs_2D_N):

        input_1D_N_1=inputs_1D_N.transpose(1,2)
        input_1D_N_1=self.BN_1D(input_1D_N_1)
        input_1D_N=input_1D_N_1.transpose(1,2)

        inputs_2D_N=inputs_2D_N.transpose(1,3)
        inputs_2D_N=self.BN_2D(inputs_2D_N)
        inputs_2D_N=inputs_2D_N.transpose(1,3)

        model_1_output = self.model_1(inputs_1D_N)
        model_2_output = self.cnn_1D(inputs_1D_N,inputs_1D_N)
        model_3_output = self.cnn_2D(inputs_2D_N,inputs_1D_N)

        output_GRU=self.output_GRU(model_1_output).view(-1,w,6)
        output_C2=self.output_C1(model_2_output).view(-1,w,6)
        output_C1=self.output_C2(model_3_output).view(-1,w,6)


        output = (output_GRU +output_C2+output_C1)/3

        return output_GRU, output_C1, output_C2, output

lr = 0.001
model = Gait_Net(12,2,100)

gait_Net = train_kinematics(train_loader, lr,30,model,path+encoder+'_gait_net_kinematics.pth')

gait_Net= Gait_Net(12,2,100)
gait_Net.load_state_dict(torch.load(path+encoder+'_gait_net_kinematics.pth'))
gait_Net.to(device)

gait_Net.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data_features_1D, data_features_2D,data_acc,data_gyr, data_targets) in enumerate(test_loader):
        output_1,output_2,output_3,output = gait_Net(data_features_1D[:,:,36:48].to(device).float(),data_features_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=data_targets

        else:
          yhat_5=torch.cat((yhat_5,output),dim=0)
          test_target=torch.cat((test_target,data_targets),dim=0)

yhat_4 = yhat_5.detach().cpu().numpy()
test_y = test_target.detach().cpu().numpy()
print(yhat_4.shape)

 ### Present ###
yhat_5=yhat_4.reshape((yhat_4.shape[0]*w,6))
test_y_r=test_y.reshape((test_y.shape[0]*w,6))

print(yhat_4.shape)

### Unpack ###
yhat_up=unpack_dataset_present(np.array(yhat_5))
test_y_up=unpack_dataset_present(np.array(test_y_r))

print(yhat_up.shape,test_y_up.shape)

### Present ###

yhat_up=yhat_up.reshape(int(len(yhat_up)/6),6)
test_y_up=test_y_up.reshape(int(len(test_y_up)/6),6)

print(yhat_up.shape,test_y_up.shape)


### Present ###

rmse,p= prediction_test(np.array(yhat_up),np.array(test_y_up))

print(rmse[0])
print(rmse[1])
print(rmse[2])

m=np.mean(rmse)

print('\n')
print(m)

print('\n')

print(p[0])
print(p[1])
print(p[2])
print('\n')

print(np.mean(p))

"""## Lightweight Model

### Training Function
"""

def train_kinematics_light(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=learn_rate)

    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data_features_1D, data_features_2D, data_acc,data_gyr, data_targets) in enumerate(train_loader):
            optimizer.zero_grad()

            target_output= model(data_acc[:,:,18:24].to(device).float(),data_gyr[:,:,18:24].to(device).float())

            loss_1=criterion(target_output, data_targets.to(device).float())

            loss_1.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_features_1D, data_features_2D,data_acc,data_gyr, data_targets in val_loader:

                output= model(data_acc[:,:,18:24].to(device).float(),data_gyr[:,:,18:24].to(device).float())
                val_loss += criterion(output, data_targets.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model

"""### Lightweight model--Korea Rehabilitation Centre collaboration"""

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

        self.fc = nn.Linear(2*2*16+16,6)

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


    def forward(self, x_acc, x_gyr):

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

class residual_net(nn.Module):
    def __init__(self, input_dim, dropout=0.10):
        super(residual_net, self).__init__()
        self.lstm_1 = nn.GRU(input_dim, 16, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(32, 8, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)

        self.fc=nn.Linear(16,2)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)
        # out_2=self.flatten(out_2)

        out=self.fc(out_2)

        return out

!pip install thop

import torch
from thop import profile

model = Kinematics_lightweight(6,6)

input = torch.randn(1, 100, 6)
flops, params = profile(model, inputs=([input,input]))

print(f"FLOPs: {flops}")



# Calculate the total size of parameters in bytes
total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f'Total size in bytes: {total_size_bytes}')

# Convert total size in bytes to kilobytes
total_size_kb = total_size_bytes / 1024
print(f'Total size in kilobytes: {total_size_kb:.2f} KB')

import torch
from thop import profile

model = residual_net(6)

input = torch.randn(1, 100, 6)
flops, params = profile(model, inputs=(input,))

print(f"FLOPs: {flops}")

# Calculate the total size of parameters in bytes
total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f'Total size in bytes: {total_size_bytes}')

# Convert total size in bytes to kilobytes
total_size_kb = total_size_bytes / 1024
print(f'Total size in kilobytes: {total_size_kb:.2f} KB')

lr = 0.001
model = Kinematics_lightweight(6,6)

gait_Net = train_kinematics_light(train_loader, lr,30,model,path+encoder+'_gait_net_kinematics_lightweight.pth')

gait_Net= Kinematics_lightweight(6,6)
gait_Net.load_state_dict(torch.load(path+encoder+'_gait_net_kinematics_lightweight.pth'))
gait_Net.to(device)

gait_Net.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data_features_1D, data_features_2D,data_acc,data_gyr, data_targets) in enumerate(test_loader):
        output = gait_Net(data_acc[:,:,18:24].to(device).float(),data_gyr[:,:,18:24].to(device).float())
        if i==0:
          yhat_5=output
          test_target=data_targets

        else:
          yhat_5=torch.cat((yhat_5,output),dim=0)
          test_target=torch.cat((test_target,data_targets),dim=0)

yhat_4 = yhat_5.detach().cpu().numpy()
test_y = test_target.detach().cpu().numpy()
print(yhat_4.shape)

 ### Present ###
yhat_5=yhat_4.reshape((yhat_4.shape[0]*w,6))
test_y_r=test_y.reshape((test_y.shape[0]*w,6))

print(yhat_4.shape)

### Unpack ###
yhat_up=unpack_dataset_present(np.array(yhat_5))
test_y_up=unpack_dataset_present(np.array(test_y_r))

print(yhat_up.shape,test_y_up.shape)

### Present ###

yhat_up=yhat_up.reshape(int(len(yhat_up)/6),6)
test_y_up=test_y_up.reshape(int(len(test_y_up)/6),6)

print(yhat_up.shape,test_y_up.shape)


### Present ###

rmse,p= prediction_test(np.array(yhat_up),np.array(test_y_up))

print(rmse[0])
print(rmse[1])
print(rmse[2])

m=np.mean(rmse)

print('\n')
print(m)

print('\n')

print(p[0])
print(p[1])
print(p[2])
print('\n')

print(np.mean(p))