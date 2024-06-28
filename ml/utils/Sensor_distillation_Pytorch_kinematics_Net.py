#Sensor_distillation_Pytorch_kinematics_Net.py 

#IMPORTS
import h5py #to read in files of data, Sanzid's data

import json #not used 

import matplotlib.pyplot as plt #this is used for the plot the graph

import numpy as np #necessary for model architecture and general processing of data

import numpy #same as above but there is inconsistent use of numpy

import statistics #not used 

#from   import loadtxt

import pandas #not directly used but later used as pd

import math #not used directly

from sklearn.preprocessing import MinMaxScaler #not used

from sklearn.metrics import mean_squared_error #used in evaluating predictions

from statistics import stdev #not used 

import time #used to measure model 

from scipy.signal import butter, filtfilt #not used

import sys #not used

from scipy.stats import randint #not used

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL #not directly used

import seaborn as sns # used for plot interactive graph. #not directly used

import itertools #not directly used

import os #not directly used

from sklearn.model_selection import train_test_split #used to split datasets

import gc #used to collect data

import torch.nn.init as init #not directly used

import torch.nn.utils.weight_norm as weight_norm #not directly used
from sklearn.preprocessing import StandardScaler #not directly used

import torch #for making, training the model and processing the data in pytorch

import torch.nn as nn #neural network construction

from torch.utils.data import TensorDataset, DataLoader #to format datasets and create data loaders

import torch.nn.functional as F #used for layer construction in model, in this case used under encoder class

from torchsummary import summary #not used directly

from torch.nn.parameter import Parameter #not used directly

import torch.optim as optim #not used directly

from tqdm import tqdm_notebook #not used directly

from google.colab import drive #for access to desired files 

############################################################################################################################################################################
############################################################################################################################################################################

drive.mount('/content/drive') #mount to access files from google drive


# %% [markdown]
# # Data Loader

# %%
##IMU, Kin-Kinematics, HoF-Histogram of Features

#Dataloader write function: retrieves data from files and combines from all trials into single np array
def data_loader_WR(subject):  
  with h5py.File('/content/drive/My Drive/Kinematics_data_10 Subjects/All_subjects_data_kinematics.h5', 'r') as hf:
    All_subjects = hf['All_subjects']
    Subject = All_subjects[subject]

    HOF=Subject['hof']
    IMU_KIN=Subject['IMU_Kin']

    treadmill_hof = HOF['Treadmill']
    levelground_hof = HOF['Levelground']
    slope_hof = HOF['Slope']
    stair_hof = HOF['Stair']
    round_hof = HOF['Round']
    obstacles_hof = HOF['Obstacles']

    treadmill_IMU_kin = IMU_KIN['Treadmill']
    levelground_IMU_kin = IMU_KIN['Levelground']
    slope_IMU_kin = IMU_KIN['Slope']
    stair_IMU_kin = IMU_KIN['Stair']
    round_IMU_kin= IMU_KIN['Round']
    obstacles_IMU_kin = IMU_KIN['Obstacles']


    hof_data=np.concatenate((treadmill_hof,levelground_hof,levelground_hof,levelground_hof,slope_hof,slope_hof,slope_hof,stair_hof),axis=0)
    IMU_kin_data=np.concatenate((treadmill_IMU_kin,levelground_IMU_kin,slope_IMU_kin,stair_IMU_kin,round_IMU_kin,obstacles_IMU_kin),axis=0)

    return np.array(hof_data), np.array(IMU_kin_data)


# %%
##IMU, Kin-Kinematics, HoF-Histogram of Features

#Dataloader function: retrieves data from files and combines from all trials into single np array
def data_loader(subject):
  with h5py.File('/content/drive/My Drive/Kinematics_data_10 Subjects/All_subjects_data_kinematics.h5', 'r') as hf:
    All_subjects = hf['All_subjects']
    Subject = All_subjects[subject]

    HOF=Subject['hof']
    IMU_KIN=Subject['IMU_Kin']

    treadmill_hof = HOF['Treadmill']
    levelground_hof = HOF['Levelground']
    slope_hof = HOF['Slope']
    stair_hof = HOF['Stair']
    round_hof = HOF['Round']
    obstacles_hof = HOF['Obstacles']

    treadmill_IMU_kin = IMU_KIN['Treadmill']
    levelground_IMU_kin = IMU_KIN['Levelground']
    slope_IMU_kin = IMU_KIN['Slope']
    stair_IMU_kin = IMU_KIN['Stair']
    round_IMU_kin= IMU_KIN['Round']
    obstacles_IMU_kin = IMU_KIN['Obstacles']


    hof_data=np.concatenate((treadmill_hof,levelground_hof,levelground_hof,levelground_hof,slope_hof,slope_hof,slope_hof,stair_hof,stair_hof,stair_hof,stair_hof),axis=0)
    IMU_kin_data=np.concatenate((treadmill_IMU_kin,levelground_IMU_kin,levelground_IMU_kin,levelground_IMU_kin,slope_IMU_kin,slope_IMU_kin,slope_IMU_kin,\
                                 stair_IMU_kin,stair_IMU_kin,stair_IMU_kin,stair_IMU_kin),axis=0)

    return np.array(hof_data), np.array(IMU_kin_data)


#uses dataloader write function to organize data from files into being separated by subject.
# %%
subject_1_data_hof, subject_1_data_IMU_Kin=data_loader_WR('Subject_1')
gc.collect()
subject_2_data_hof, subject_2_data_IMU_Kin=data_loader_WR('Subject_2')
gc.collect()
subject_3_data_hof, subject_3_data_IMU_Kin=data_loader_WR('Subject_3')
gc.collect()
subject_4_data_hof, subject_4_data_IMU_Kin=data_loader_WR('Subject_4')
gc.collect()
subject_5_data_hof, subject_5_data_IMU_Kin=data_loader_WR('Subject_5')
gc.collect()
subject_6_data_hof, subject_6_data_IMU_Kin=data_loader_WR('Subject_6')
gc.collect()
subject_7_data_hof, subject_7_data_IMU_Kin=data_loader_WR('Subject_7')
gc.collect()
subject_8_data_hof, subject_8_data_IMU_Kin=data_loader_WR('Subject_8')
gc.collect()
subject_9_data_hof, subject_9_data_IMU_Kin=data_loader_WR('Subject_9')
gc.collect()
subject_10_data_hof, subject_10_data_IMU_Kin=data_loader_WR('Subject_10')
gc.collect()

# %% [markdown]
# # Subject Selection

#create directory that has subject 1 data
# %%
main_dir = "/content/drive/My Drive/public dataset/Public_dataset_2/Subject01"
# os.mkdir(main_dir)
path="/content/"
subject='Subject_01'
encoder='lstm'

#LOOSCV 
# %%
train_data_hof=np.concatenate((subject_1_data_hof,subject_2_data_hof,subject_3_data_hof,subject_4_data_hof,subject_5_data_hof,
                               subject_6_data_hof,subject_8_data_hof,subject_9_data_hof,subject_10_data_hof),axis=0)

train_data_IMU_Kin=np.concatenate((subject_1_data_IMU_Kin,subject_2_data_IMU_Kin,subject_3_data_IMU_Kin,subject_4_data_IMU_Kin,subject_5_data_IMU_Kin,
                               subject_6_data_IMU_Kin,subject_8_data_IMU_Kin,subject_9_data_IMU_Kin,subject_10_data_IMU_Kin),axis=0)


test_data_hof=subject_7_data_hof
test_data_IMU_Kin=subject_7_data_IMU_Kin

#define some model parameters
# %%
alpha=0.50
num_epoch=30

# %% [markdown]
# # Data Processing

# %%
##### IMUs-0:48
# Sensor 1- Sternum
# Sensor 2-Sacrum
# Sensor 3-R_thigh
# Sensor 4-L_thigh
# Sensor 5-R_shank
# Sensor 6-L_shank
# Sensor 7-R_dorsal
# Sensor 8-L_dorsal

#train_X and train_Y defined
train_dataset_IMU=train_data_IMU_Kin[:,0:48]
train_dataset_hof=train_data_hof
train_dataset_target=np.concatenate((train_data_IMU_Kin[:,55:56],train_data_IMU_Kin[:,58:60],train_data_IMU_Kin[:,62:63],train_data_IMU_Kin[:,65:67]),axis=1) ## Left and right leg hip, knee,ankle angle

#test_X and test_Y defined
test_dataset_IMU=test_data_IMU_Kin[:,0:48]
test_dataset_hof=test_data_hof
test_dataset_target=np.concatenate((test_data_IMU_Kin[:,55:56],test_data_IMU_Kin[:,58:60],test_data_IMU_Kin[:,62:63],test_data_IMU_Kin[:,65:67]),axis=1)

#check shape
print(train_dataset_IMU.shape)
print(train_dataset_hof.shape)
print(train_dataset_target.shape)

#collect the defined data
gc.collect()
gc.collect()
gc.collect()


# %% [markdown]
# # Data creation

# %%
## Creating a dataset with overlapping window of 100 samples with overlap of 50 samples ##

# # convert an array of values into a dataset matrix
def create_dataset_present(dataset_1, window=100):
  dataX= []
  k=0
  shift=50
  for i in range(int(len(dataset_1)/shift)-1):
    j=shift*k
    a = dataset_1[j:j+window,:]
    dataX.append(a)
    k=k+1
  return np.array(dataX)

#must be a saving function?
# %%
import gc
gc.collect()

#creating the test and train datasets with a w parameter that I guess is for windowing
# %%
### Reconstruction/Present Dataset ###
w=100

train_X_3=create_dataset_present(train_dataset_IMU,w)
train_y_3=create_dataset_present(train_dataset_target,w)

test_X_1D=create_dataset_present(test_dataset_IMU,w)
test_y=create_dataset_present(test_dataset_target,w)

#reshape the datasets
# %%
train_y_3=train_y_3.reshape(train_y_3.shape[0],w,6)
test_y=test_y.reshape(test_y.shape[0],w,6)


#split into training and validation data
# %%
train_X_1D, X_validation_1D, train_y_5, Y_validation = train_test_split(train_X_3,train_y_3, test_size=0.20, random_state=True)

print(train_X_1D.shape,train_y_5.shape,X_validation_1D.shape,Y_validation.shape)

#define amount of features and reshape input data accordingly
# %%
features=6

train_X_2D=train_X_1D.reshape(train_X_1D.shape[0],train_X_1D.shape[1],features,8)
test_X_2D=test_X_1D.reshape(test_X_1D.shape[0],test_X_1D.shape[1],features,8)
X_validation_2D= X_validation_1D.reshape(X_validation_1D.shape[0],
                                                   X_validation_1D.shape[1],features,8)


print(train_X_2D.shape,test_X_2D.shape,X_validation_2D.shape)

# %% [markdown]
# # Different Function for models

#prediction function
# %%
def prediction_test(yhat,test_y_up):

    test_o=test_y_up
    yhat=yhat

    y_1=yhat[:,0]
    y_2=yhat[:,1]
    y_3=yhat[:,2]
    y_4=yhat[:,3]
    y_5=yhat[:,4]
    y_6=yhat[:,5]


    y_test_1=test_o[:,0]
    y_test_2=test_o[:,1]
    y_test_3=test_o[:,2]
    y_test_4=test_o[:,3]
    y_test_5=test_o[:,4]
    y_test_6=test_o[:,5]


    ###calculate RMSE

    rmse_1 =np.sqrt(mean_squared_error(y_test_1,y_1))
    rmse_2 =np.sqrt(mean_squared_error(y_test_2,y_2))
    rmse_3 =np.sqrt(mean_squared_error(y_test_3,y_3))
    rmse_4 =np.sqrt(mean_squared_error(y_test_4,y_4))
    rmse_5 =np.sqrt(mean_squared_error(y_test_5,y_5))
    rmse_6 =np.sqrt(mean_squared_error(y_test_6,y_6))


    p_1=np.corrcoef(y_1, y_test_1)[0, 1]
    p_2=np.corrcoef(y_2, y_test_2)[0, 1]
    p_3=np.corrcoef(y_3, y_test_3)[0, 1]
    p_4=np.corrcoef(y_4, y_test_4)[0, 1]
    p_5=np.corrcoef(y_5, y_test_5)[0, 1]
    p_6=np.corrcoef(y_6, y_test_6)[0, 1]

    ### Getiing single RMSE and PCC value for a joint
    p=np.array([(p_1+p_4)/2,(p_2+p_5)/2,(p_3+p_6)/2])

    rmse=np.array([(rmse_1+rmse_4)/2,(rmse_2+rmse_5)/2,(rmse_3+rmse_6)/2])



    return rmse,p

# %%
## Evaluting data with the original form without any overlaps

# # convert an array of values into a dataset matrix
def unpack_dataset_present(dataset_1):
  dataX= []
  k=1
  l=0
  shift=100
  for i in range(int(len(dataset_1)/shift)-1):
    j=shift*k
    a = dataset_1[l:j,:]
    l=0
    l=j+50
    dataX=np.append(dataX,a)
    k=k+1
    j=0
  return np.array(dataX)

# %% [markdown]
# # Data Preparation

# %%
### IMUs- Chest, Waist, Right Foot, Right shank, Right thigh, Left Foot, Left shank, Left thigh, 2D-body coordinate
### 0:48- IMU, 48:92-2D body coordinate, 92:97-- Target


### Data Processing

#define batch size and convert to tensors
batch_size = 64

train_features_1D = torch.Tensor(train_X_1D)
val_features_1D = torch.Tensor(X_validation_1D)
test_features_1D = torch.Tensor(test_X_1D)

train_features_2D = torch.Tensor(train_X_2D)
val_features_2D = torch.Tensor(X_validation_2D)
test_features_2D = torch.Tensor(test_X_2D)

train_targets = torch.Tensor(train_y_5)
val_targets = torch.Tensor(Y_validation)
test_targets = torch.Tensor(test_y)

#define train, val and test datasets
## all Modality Features
train = TensorDataset(train_features_1D, train_features_2D, train_targets)
val = TensorDataset(val_features_1D, val_features_2D,val_targets)
test = TensorDataset(test_features_1D,test_features_2D,test_targets)

#define dataloaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)


# %% [markdown]
# # Important Functions

#loss function
# %%
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        mse = nn.MSELoss()(pred, target)
        rmse = torch.sqrt(mse)
        return rmse

#coefficient loss function
# %%
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

#define device
# %%
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %% [markdown]
# # Training Loop Function

#train function
# %%
def train_kinematics(train_loader, learn_rate, EPOCHS, model,filename,k_1,k_2,k_3,k_4):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()
    # criterion =correlation_coefficient_loss_joint_pytorch()

    # criterion=PearsonCorrLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    optimizer_1 = torch.optim.Adam(model.model_1.parameters(), lr=learn_rate)
    optimizer_2 = torch.optim.Adam(model.cnn_1D.parameters(), lr=learn_rate)
    optimizer_3 = torch.optim.Adam(model.cnn_2D.parameters(), lr=learn_rate)

    optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # model.model_1.train()
        # model.cnn_1D.train()
        # model.cnn_2D.train()
        model.train()

        for i, (data_features_1D, data_features_2D, data_targets) in enumerate(train_loader):


# ###################################################################################################################################

            # optimizer_1.zero_grad()

            # output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            # loss_1=criterion(output_1, data_targets.to(device).float())

            # loss_1.backward()
            # optimizer_1.step()

# ###################################################################################################################################

            # optimizer_2.zero_grad()

            # output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            # loss_2=criterion(output_2, data_targets.to(device).float())

            # loss_2.backward()
            # optimizer_2.step()

# ###################################################################################################################################


            # optimizer_3.zero_grad()

            # output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            # loss_3=criterion(output_3, data_targets.to(device).float())

            # loss_3.backward()
            # optimizer_3.step()

# ###################################################################################################################################



            optimizer.zero_grad()

            output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            loss=criterion(output, data_targets.to(device).float())

            loss.backward()
            optimizer.step()


###################################################################################################################################




            # Compute the regularization loss for the custom linear layers
            # regularization_loss = 0.0
            # if hasattr(model.output_GRU, 'regularizer_loss'):
            #     regularization_loss += model.output_GRU.regularizer_loss()
            # if hasattr(model.output_C1, 'regularizer_loss'):
            #     regularization_loss += model.output_C1.regularizer_loss()
            # if hasattr(model.output_C2, 'regularizer_loss'):
            #     regularization_loss += model.output_C2.regularizer_loss()

            # loss=criterion(output_1, data_targets.to(device).float())+criterion(output_2, data_targets.to(device).float())\
            # +criterion(output_3, data_targets.to(device).float())+criterion(output, data_targets.to(device).float())

            loss_1=criterion(output, data_targets.to(device).float())

            # loss.backward()
            # optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_features_1D, data_features_2D, data_targets in val_loader:
                output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
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

# %% [markdown]
# # Model

# %%
# class RegularizedLinear(nn.Module):
#     def __init__(self, in_features, out_features, weight_decay=0.001):
#         super(RegularizedLinear, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.weight_decay = weight_decay

#     def forward(self, input):
#         return self.linear(input)

#     def regularizer_loss(self):
#         return self.weight_decay * torch.sum(self.linear.bias**2)


#encoder part of final model
# %%
class Encoder_GRU(nn.Module):
    def __init__(self, input_shape_1D, input_shape_2D, dropout):
        super(Encoder_GRU, self).__init__()
        self.lstm_1 = nn.GRU(input_shape_1D, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)
        self.output_GRU = nn.Linear(w*128, 6 * w)

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)
        out_2=self.flatten(out_2)

        output_GRU=self.output_GRU(out_2).view(-1,w,6)

        return output_GRU

#encoder part of final model
# %%
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

#encoder part of final model
# %%
class Encoder_CNN_1D(nn.Module):
    def __init__(self, input_shape_1D, input_shape_2D, dropout, hidden_dim=64, output_size=128, kernel_size=3, stride=1, padding=1):
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

    def forward(self, x, x_1):

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

#encoder part of final model
# %%
class Encoder_CNN_2D(nn.Module):
    def __init__(self, input_shape_1D,input_shape_2D, dropout, hidden_dim=64, output_size=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
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

    def forward(self, x_1,x):

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

# %% [markdown]
# # Teacher Model

#model that puts everything together
# %%
class teacher(nn.Module):
    def __init__(self, input_shape_1D,input_shape_2D, w):
        super(teacher, self).__init__()
        self.w = w

        # 1D Models

        self.model_1=Encoder_GRU(input_shape_1D,input_shape_2D,0.40)
        self.cnn_1D = Encoder_CNN_1D(input_shape_1D,input_shape_2D,0.40)
        self.cnn_2D = Encoder_CNN_2D(input_shape_1D,input_shape_2D,0.40)

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)
        self.BN_2D= nn.BatchNorm2d(input_shape_2D, affine=False)


    def forward(self, inputs_1D_N, inputs_2D_N):

        # input_1D_N_1=inputs_1D_N.transpose(1,2)
        # input_1D_N_1=self.BN_1D(input_1D_N_1)
        # input_1D_N=input_1D_N_1.transpose(1,2)

        # inputs_2D_N=inputs_2D_N.transpose(1,3)
        # inputs_2D_N=self.BN_2D(inputs_2D_N)
        # inputs_2D_N=inputs_2D_N.transpose(1,3)

        output_GRU = self.model_1(inputs_1D_N)
        output_C2 = self.cnn_1D(inputs_1D_N,inputs_1D_N)
        output_C1 = self.cnn_2D(inputs_1D_N,inputs_2D_N)


        output = (output_GRU +output_C2+output_C1)/3

        return output_GRU, output_C1, output_C2, output

#run model initialization and train model, teacher model
# %%
k_1=0
k_2=48
k_3=0
k_4=8

lr = 0.001
model = teacher(k_2-k_1,k_4-k_3,100)

gait_Net_teacher = train_kinematics(train_loader, lr,num_epoch,model,path+encoder+'_teacher.pth',k_1, k_2, k_3, k_4)

#use train model to perform evaluation
# %%
gait_Net_teacher= teacher(k_2-k_1,k_4-k_3,100)
gait_Net_teacher.load_state_dict(torch.load(path+encoder+'_teacher.pth'))
gait_Net_teacher.to(device)

gait_Net_teacher.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data_features_1D, data_features_2D, data_targets) in enumerate(test_loader):
        output_1,output_2,output_3,output = gait_Net_teacher(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
        if i==0:
          yhat_5=output
          test_target=data_targets

        else:
          yhat_5=torch.cat((yhat_5,output),dim=0)
          test_target=torch.cat((test_target,data_targets),dim=0)



#yhat variables = output, and test variables = test
yhat_4 = yhat_5.detach().cpu(). ()
test_y = test_target.detach().cpu(). ()
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

print(yhat_up.shape)


### Present ###

#predictions, rmse
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

ablation_1=np.hstack([rmse,p])

# %% [markdown]
# # Student Model

#student model
# %%
class student(nn.Module):
    def __init__(self, input_shape_1D,input_shape_2D, w):
        super(student, self).__init__()
        self.w = w

        # 1D Models

        self.model_1=Encoder_GRU(input_shape_1D,input_shape_2D,0.40)
        self.cnn_1D = Encoder_CNN_1D(input_shape_1D,input_shape_2D,0.40)
        self.cnn_2D = Encoder_CNN_2D(input_shape_1D,input_shape_2D,0.40)

        self.BN_1D= nn.BatchNorm1d(input_shape_1D, affine=False)
        self.BN_2D= nn.BatchNorm2d(input_shape_2D, affine=False)


    def forward(self, inputs_1D_N, inputs_2D_N):

        # input_1D_N_1=inputs_1D_N.transpose(1,2)
        # input_1D_N_1=self.BN_1D(input_1D_N_1)
        # input_1D_N=input_1D_N_1.transpose(1,2)

        # inputs_2D_N=inputs_2D_N.transpose(1,3)
        # inputs_2D_N=self.BN_2D(inputs_2D_N)
        # inputs_2D_N=inputs_2D_N.transpose(1,3)

        output_GRU = self.model_1(inputs_1D_N)
        output_C2 = self.cnn_1D(inputs_1D_N,inputs_1D_N)
        output_C1 = self.cnn_2D(inputs_1D_N,inputs_2D_N)


        output = (output_GRU +output_C2+output_C1)/3

        return output_GRU, output_C1, output_C2, output

#model initialization and training, student model 
# %%
k_1=36
k_2=48
k_3=6
k_4=8

lr = 0.001
model = student(k_2-k_1,k_4-k_3,100)

gait_Net_student = train_kinematics(train_loader, lr,num_epoch,model,path+encoder+'_student.pth',k_1, k_2, k_3, k_4)

#to device and eval
# %%
gait_Net_student= student(k_2-k_1,k_4-k_3,100)
gait_Net_student.load_state_dict(torch.load(path+encoder+'_student.pth'))
gait_Net_student.to(device)

gait_Net_student.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data_features_1D, data_features_2D, data_targets) in enumerate(test_loader):
        output_1,output_2,output_3,output = gait_Net_student(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
        if i==0:
          yhat_5=output
          test_target=data_targets

        else:
          yhat_5=torch.cat((yhat_5,output),dim=0)
          test_target=torch.cat((test_target,data_targets),dim=0)



#yhat variables represents output from model and test variables represent the targets
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

print(yhat_up.shape)


### Present ###

rmse,p= prediction_test(np.array(yhat_up),np.array(test_y_up))

#rmse array
print(rmse[0])
print(rmse[1])
print(rmse[2])

#mean rmse 
m=np.mean(rmse)

print('\n')

#mean rmse
print(m)

print('\n')

#prediction array
print(p[0])
print(p[1])
print(p[2])
print('\n')

#mean prediction
print(np.mean(p))

#ablation_2 defined, holds rmse and predictions
ablation_2=np.hstack([rmse,p])

# %% [markdown]
# # Sensor Distillation

#definition to train sensor distillation model, uses student model, evaluate teacher and student model on loss
# %%
def train_SD(train_loader, learn_rate, EPOCHS, student, teacher, filename, k_1s, k_2s, k_3s, k_4s, k_1t, k_2t, k_3t, k_4t):

    if torch.cuda.is_available():
      student.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()
    # criterion =correlation_coefficient_loss_joint_pytorch()

    # criterion=PearsonCorrLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=learn_rate)
    # optimizer_1 = torch.optim.Adam(student.model_1.parameters(), lr=learn_rate)
    # optimizer_2 = torch.optim.Adam(student.cnn_1D.parameters(), lr=learn_rate)
    # optimizer_3 = torch.optim.Adam(student.cnn_2D.parameters(), lr=learn_rate)

    # optimizer_t = torch.optim.Adam(teacher.parameters(), lr=learn_rate)

    # optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # student.model_1.train()
        # student.cnn_1D.train()
        # student.cnn_2D.train()
        student.train()

        for i, (data_features_1D, data_features_2D, data_targets) in enumerate(train_loader):



            # with torch.no_grad():
            #  output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())



# ###################################################################################################################################

            # optimizer_1.zero_grad()

            # output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            # loss_1=criterion(output_1, data_targets.to(device).float())+alpha*criterion(output_1, output_1t)

            # loss_1.backward()
            # optimizer_1.step()

# ###################################################################################################################################

            # optimizer_2.zero_grad()

            # output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            # # with torch.no_grad():
            # #  output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())

            # loss_2=criterion(output_2, data_targets.to(device).float())+alpha*criterion(output_2, output_2t)

            # loss_2.backward()
            # optimizer_2.step()

# ###################################################################################################################################


            # optimizer_3.zero_grad()

            # output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            # # with torch.no_grad():
            # #  output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())

            # loss_3=criterion(output_3, data_targets.to(device).float())+alpha*criterion(output_3, output_3t)

            # loss_3.backward()
            # optimizer_3.step()

# ###################################################################################################################################



            optimizer.zero_grad()

            output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            with torch.no_grad():
             output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())

            loss=criterion(output, data_targets.to(device).float())+alpha*criterion(output, outputt)

            loss.backward()
            optimizer.step()


###################################################################################################################################




            # Compute the regularization loss for the custom linear layers
            # regularization_loss = 0.0
            # if hasattr(model.output_GRU, 'regularizer_loss'):
            #     regularization_loss += model.output_GRU.regularizer_loss()
            # if hasattr(model.output_C1, 'regularizer_loss'):
            #     regularization_loss += model.output_C1.regularizer_loss()
            # if hasattr(model.output_C2, 'regularizer_loss'):
            #     regularization_loss += model.output_C2.regularizer_loss()

            # loss=criterion(output_1, data_targets.to(device).float())+criterion(output_2, data_targets.to(device).float())\
            # +criterion(output_3, data_targets.to(device).float())+criterion(output, data_targets.to(device).float())

            loss_1=criterion(output, data_targets.to(device).float())

            # loss.backward()
            # optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_features_1D, data_features_2D, data_targets in val_loader:
                output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())
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
            torch.save(student.state_dict(), filename)
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



    return student

#student and teacher model initialization and then train sensor distillation using both initialized models 
# %%
k_1s=36
k_2s=48
k_3s=6
k_4s=8

lr = 0.001
Student = student(k_2s-k_1s,k_4s-k_3s,100)

k_1t=0
k_2t=48
k_3t=0
k_4t=8

lr = 0.001
Teacher = teacher(k_2t-k_1t,k_4t-k_3t,100)
Teacher.load_state_dict(torch.load(path+encoder+'_teacher.pth'))
Teacher.to(device)



student_KD= train_SD(train_loader, lr,num_epoch, Student,Teacher, path+'student_kd.pth', k_1s, k_2s, k_3s, k_4s, k_1t, k_2t, k_3t, k_4t)

#sensor distillation eval using student model
# %%
gait_Net_student_SD= student(k_2s-k_1s,k_4s-k_3s,100)
gait_Net_student_SD.load_state_dict(torch.load(path+'student_kd.pth'))
gait_Net_student_SD.to(device)

gait_Net_student_SD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data_features_1D, data_features_2D, data_targets) in enumerate(test_loader):
        output_1,output_2,output_3,output = gait_Net_student_SD(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())
        if i==0:
          yhat_5=output
          test_target=data_targets

        else:
          yhat_5=torch.cat((yhat_5,output),dim=0)
          test_target=torch.cat((test_target,data_targets),dim=0)



#yhat variables represent output and test variables here represent targets
yhat_4 = yhat_5.detach().cpu().numpy()
test_y = test_target.detach().cpu().numpy()
print(yhat_4.shape)

 ### Present ###
yhat_5=yhat_4.reshape((yhat_4.shape[0]*w,6))
test_y_r=test_y.reshape((test_y.shape[0]*w,6))

print(yhat_4.shape)

### Unpack ###
#yhat_up and test_y_up
yhat_up=unpack_dataset_present(np.array(yhat_5))
test_y_up=unpack_dataset_present(np.array(test_y_r))

print(yhat_up.shape,test_y_up.shape)

### Present ###

yhat_up=yhat_up.reshape(int(len(yhat_up)/6),6)
test_y_up=test_y_up.reshape(int(len(test_y_up)/6),6)

print(yhat_up.shape,test_y_up.shape)

print(yhat_up.shape)


### Present ###

#rmse and prediction, testing yhat output and test targets
rmse,p= prediction_test(np.array(yhat_up),np.array(test_y_up))

#rmse array
print(rmse[0])
print(rmse[1])
print(rmse[2])

m=np.mean(rmse)

print('\n')

#mean rmse
print(m)

print('\n')

#prediction array
print(p[0])
print(p[1])
print(p[2])
print('\n')

#mean prediction
print(np.mean(p))

#ablation 3 defined with rmse resukt and prediction array
ablation_3=np.hstack([rmse,p])


