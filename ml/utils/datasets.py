import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from data_utils import get_data_from_wav_file
from torchvision.transforms import ToTensor
import torchaudio.transforms as T
from scipy.io import wavfile

#Dataset class for IMU and JOINT 
class ImuJointPairDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.root_dir_train = os.path.join(self.config.dataset_root, self.config.dataset_train_name)
        self.root_dir_test = os.path.join(self.config.dataset_root, self.config.dataset_test_name)
        
        #define access to training and testing windowed datasets
        full_data_train = pd.read_csv(self.root_dir_train + "_info.csv")
        full_data_test = pd.read_csv(self.root_dir_test + "_info.csv")
        
        #access to specific dataset        
        if self.split == 'train':
            self.data = full_data_train
        elif self.split == 'test':
            self.data = full_data_test
        else:
            raise ValueError("Invalid input. Expected 'train' or 'test'.")

    #return length of data
    def __len__(self):
        return len(self.data)
    
    #function to grab imu or joint data specifically from entire dataset
    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
            #idx = idx.tolist()
        
        #access to either train or test dataset
        if self.split=="train":
            file_path = os.path.join(self.root_dir_train, self.data.iloc[idx, 1])
        if self.split=="test":
            file_path = os.path.join(self.root_dir_test, self.data.iloc[idx, 1])
            
        #combined imu and joint data
        combined_data, _ = get_data_from_wav_file(file_path)
        
        #IMU
        imu_data = combined_data[:, :self.config.num_channels_imu]
        
        #JOINT
        joint_data = combined_data[:, self.config.num_channels_joints:]
        
        #Apply transformations
        imu_data = self.apply_transforms(imu_data, self.config.imu_transforms)
        joint_data = self.apply_transforms(joint_data, self.config.joint_transforms)

        return imu_data, joint_data
        
    #function to apply transforms
    def apply_transforms(self, data, transforms):
        for transform in transforms:
            data = transform(data)

        data = ToTensor()(data).squeeze(0)
        return data
