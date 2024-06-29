import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_utils import get_data_from_wav_file
from torchvision.transforms import ToTensor

class ImuJointPairDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.input_format = config.input_format
        self.channels_imu = config.channels_imu
        self.channels_joints = config.channels_joints

        # Ensure the dataset name includes the window length, overlap, and num patients
        dataset_name = f"{self.config.dataset_name}_wl{self.config.window_length}_ol{self.config.window_overlap}_np{self.config.num_patients}"
        self.root_dir_train = os.path.join(self.config.dataset_root, dataset_name, self.config.dataset_train_name)
        self.root_dir_test = os.path.join(self.config.dataset_root, dataset_name, self.config.dataset_test_name)

        # Load the info CSV files for the training and testing datasets
        train_info_path = os.path.join(self.config.dataset_root, dataset_name, "train_info.csv")
        test_info_path = os.path.join(self.config.dataset_root, dataset_name, "test_info.csv")

        self.data = pd.read_csv(train_info_path) if split == 'train' else pd.read_csv(test_info_path)

    # Return length of data
    def __len__(self):
        return len(self.data)

    # Function to grab IMU or joint data specifically from the entire dataset
    def __getitem__(self, idx):
        # Access to either train or test dataset
        if self.split == "train":
            file_path = os.path.join(self.root_dir_train, self.data.iloc[idx, 0])
        else:
            file_path = os.path.join(self.root_dir_test, self.data.iloc[idx, 0])

        # Check if the input format is CSV or WAV and load the data accordingly
        if self.input_format == "wav":
            combined_data, _ = get_data_from_wav_file(file_path)
        elif self.input_format == "csv":
            combined_data = pd.read_csv(file_path).values
        else:
            raise ValueError("Unsupported input format: {}".format(self.input_format))

        # IMU
        imu_data = combined_data[:, self.channels_imu]

        # JOINT
        joint_data = combined_data[:, self.channels_joints]

        # Apply transformations
        imu_data = self.apply_transforms(imu_data, self.config.imu_transforms)
        joint_data = self.apply_transforms(joint_data, self.config.joint_transforms)

        return imu_data, joint_data

    # Function to apply transforms
    def apply_transforms(self, data, transforms):
        for transform in transforms:
            data = transform(data)

        data = ToTensor()(data).squeeze(0)
        return data

if __name__ == '__main__':
    # Test dataset
    from configs import config_general
    config = config_general(
        data_folder_name="../../datacollection/vicon",
        dataset_root="../../datasets",
        dataset_name="two_subject",
        window_length=100,
        window_overlap=50,
        input_format="csv",
        num_patients=2,
        channels_imu=range(0, 6),
        channels_joints=range(6, 12),
        imu_transforms=[],
        joint_transforms=[]
    )
    dataset = ImuJointPairDataset(config, split='train')
    print(dataset.__getitem__(0))
