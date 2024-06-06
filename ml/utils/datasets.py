import os
import pandas as pd
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from data_utils import get_data_from_wav_file
import torch
from torchvision.transforms import ToTensor

class EmgJointPairDataset(Dataset):
    def __init__(self, config, split='train'):
        """
        Initializes the dataset with a train/test split.
        :param config: Configuration object with dataset parameters.
        :param split: 'train' or 'test' to specify the subset of data.
        :param test_size: The proportion of the dataset to include in the test split.
        """
        self.config = config
        self.split = split
        self.root_dir = os.path.join(self.config.dataset_root, self.config.dataset_name)
        full_data = pd.read_csv(self.root_dir + "_info.csv")
        
        # Splitting the dataset into training and testing
        train_data, test_data = train_test_split(full_data, test_size=1-self.config.train_test_split, random_state=self.config.seed)
        
        if self.split == 'train':
            self.data = train_data
        elif self.split == 'test':
            self.data = test_data
        else:
            raise ValueError("Invalid split. Expected 'train' or 'test'.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
            #idx = idx.tolist()
        
        file_path = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        combined_data, _ = get_data_from_wav_file(file_path)
        #emg_data = combined_data[:, :self.config.num_channels_emg]
        emg_data = combined_data[ :self.config.num_channels_emg]
        #joint_data = combined_data[:, self.config.num_channels_emg:]
        joint_data = combined_data[ self.config.num_channels_emg:]
        
        emg_data = self.apply_transforms(emg_data, self.config.emg_transforms)
        joint_data = self.apply_transforms(joint_data, self.config.joint_transforms)

        return emg_data, joint_data
        
    def apply_transforms(self, data, transforms):
        for transform in transforms:
            data = transform(data)

        data = ToTensor()(data)
        return data
