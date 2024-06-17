import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from data_utils import get_data_from_wav_file
from torchvision.transforms import ToTensor
import torchaudio.transforms as T
from scipy.io import wavfile

class ImuJointPairDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.root_dir_train = os.path.join(self.config.dataset_root, self.config.dataset_train_name)
        self.root_dir_test = os.path.join(self.config.dataset_root, self.config.dataset_test_name)
        full_data_train = pd.read_csv(self.root_dir_train + "_info.csv")
        full_data_test = pd.read_csv(self.root_dir_test + "_info.csv")
        
        # Splitting the dataset into training and testing

        
        if self.split == 'train':
            self.data = full_data_train
        elif self.split == 'test':
            self.data = full_data_test
        else:
            raise ValueError("Invalid split. Expected 'train' or 'test'.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx, split):
        #if torch.is_tensor(idx):
            #idx = idx.tolist()
        if split=="train":
            file_path = os.path.join(self.root_dir_train, self.data.iloc[idx, 1])
        if split=="test":
            file_path = os.path.join(self.root_dir_test, self.data.iloc[idx, 1])
            
        combined_data, _ = get_data_from_wav_file(file_path)
        emg_data = combined_data[:, :self.config.num_channels_emg]
        joint_data = combined_data[:, self.config.num_channels_emg:]
        
        emg_data = self.apply_transforms(emg_data, self.config.emg_transforms)
        joint_data = self.apply_transforms(joint_data, self.config.joint_transforms)

        return emg_data, joint_data
        
    def apply_transforms(self, data, transforms):
        for transform in transforms:
            data = transform(data)

        data = ToTensor()(data)
        return data
