import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import ToTensor
from data_utils import get_data_from_wav_file
import joblib

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

        # Define the path for saving the scaler
        self.scaler_save_path = os.path.join(self.config.dataset_root, dataset_name, "scaler.pkl")

        # Initialize and fit/load the scaler using the training data
        if os.path.exists(self.scaler_save_path):
            self.scaler = joblib.load(self.scaler_save_path)
        elif split == 'train':
            self.scaler = StandardScaler()
            self._fit_scaler(train_info_path)
        else:
            raise FileNotFoundError("Scaler not found. Ensure you run the training split first to fit and save the scaler.")

    def _fit_scaler(self, train_info_path):
        all_data = []
        train_info = pd.read_csv(train_info_path)
        for idx in range(len(train_info)):
            file_path = os.path.join(self.root_dir_train, train_info.iloc[idx, 0])
            if self.input_format == "csv":
                combined_data = pd.read_csv(file_path)
                imu_data = combined_data[self.channels_imu].values
                joint_data = combined_data[self.channels_joints].values
                all_data.append(np.concatenate([imu_data, joint_data], axis=1))  # Concatenate along axis 1 (channels)
        all_data = np.vstack(all_data)  # Stack along axis 0 (time)
        self.scaler.fit(all_data)

        # Save the scaler for later use
        os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_save_path)

    def __len__(self):
        return len(self.data)

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
            combined_data = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported input format: {}".format(self.input_format))
        
        # IMU
        if isinstance(self.channels_imu, slice):
            imu_data = combined_data.iloc[:, self.channels_imu].values if self.input_format == "csv" else combined_data[:, self.channels_imu]
        else:
            imu_data = combined_data[self.channels_imu].values if self.input_format == "csv" else combined_data[:, self.channels_imu]

        # JOINT
        if isinstance(self.channels_joints, slice):
            joint_data = combined_data.iloc[:, self.channels_joints].values if self.input_format == "csv" else combined_data[:, self.channels_joints]
        else:
            joint_data = combined_data[self.channels_joints].values if self.input_format == "csv" else combined_data[:, self.channels_joints]

        # Combine IMU and Joint data for scaling
        combined_data = np.concatenate([imu_data, joint_data], axis=1)  # Concatenate along axis 1 (channels)
        scaled_data = self.scaler.transform(combined_data)

        # Separate the scaled data back into IMU and Joint
        imu_data = scaled_data[:, :imu_data.shape[1]]
        joint_data = scaled_data[:, imu_data.shape[1]:]

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
        channels_imu=['ACCX2', 'ACCY2', 'ACCZ2', 'GYROX2', 'GYROY2', 'GYROZ2'],
        channels_joints=['elbow_flex_r'],
        imu_transforms=[],
        joint_transforms=[]
    )

    dataset = ImuJointPairDataset(config, split='train')
    print(dataset.__getitem__(0))
    dataset = ImuJointPairDataset(config, split='test')
    print(dataset.__getitem__(0))
