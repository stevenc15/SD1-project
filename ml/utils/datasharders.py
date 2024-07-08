import os
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from data_utils import get_data_from_wav_file
from tqdm import tqdm

class DataSharder:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.input_format = config.input_format
        self.data_folder_path = config.data_folder_name
        self.window_length = int(config.window_length)  # Ensure window_size is an integer
        self.window_overlap = int(config.window_overlap)  # Ensure overlap is an integer
        self.num_patients = int(config.num_patients)  # Ensure num_patients is an integer

    def load_data(self):
        patient_folders_list = [f for f in os.listdir(self.data_folder_path) if os.path.isdir(os.path.join(self.data_folder_path, f))]
        training_patients = patient_folders_list[:self.num_patients-1]
        testing_patients = patient_folders_list[self.num_patients-1:]

        if self.input_format == 'wav':
            self._process_and_save_patients_wav(training_patients, "train")
            self._process_and_save_patients_wav(testing_patients, "test")
        elif self.input_format == 'csv':
            self._process_and_save_patients_csv(training_patients, "train")
            self._process_and_save_patients_csv(testing_patients, "test")
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

    def _process_and_save_patients_wav(self, patient_id_list, split):
        for patient_id in tqdm(patient_id_list, desc=f"Processing {split} patients"):
            for session_index in tqdm(range(self.config.num_sessions), desc=f"Processing sessions for {patient_id}", leave=False):
                imu_data, imu_sample_rate = self._load_wav_file(patient_id, session_index, "IMU")
                joints_data, joints_sample_rate = self._load_wav_file(patient_id, session_index, "JOINTS")

                imu_data = self._resample_data(imu_data, imu_sample_rate)
                joints_data = self._resample_data(joints_data, joints_sample_rate)

                combined_data = torch.cat((imu_data, joints_data), dim=1)
                self._save_windowed_data(combined_data, patient_id, session_index, split)

    def _load_wav_file(self, patient_id, session_index, file_type):
        file_path = os.path.join(self.data_folder_path, patient_id, f"run{session_index}_{file_type}.wav")
        data, sample_rate = get_data_from_wav_file(file_path)
        return torch.tensor(data, dtype=torch.float32), sample_rate

    def _resample_data(self, data, sample_rate):
        if sample_rate != self.sample_rate:
            data = T.Resample(sample_rate, self.sample_rate)(data.T).T
        return data

    def _process_and_save_patients_csv(self, patient_id_list, split):
        for patient_id in tqdm(patient_id_list, desc=f"Processing {split} patients"):
            patient_files = os.listdir(os.path.join(self.data_folder_path, patient_id, "combined"))
            for session_file in tqdm(patient_files, desc=f"Processing sessions for {patient_id}", leave=False):
                data = pd.read_csv(os.path.join(self.data_folder_path, patient_id, "combined", session_file))
                self._save_windowed_data(data, patient_id, session_file.split('.')[0], split, is_csv=True)

    def _save_windowed_data(self, data, patient_id, session_id, split, is_csv=False):
        dataset_name = f"{self.config.dataset_name}_wl{self.window_length}_ol{self.window_overlap}_np{self.num_patients}"
        dataset_folder = os.path.join(self.config.dataset_root, dataset_name, self.config.dataset_train_name if split == "train" else self.config.dataset_test_name)
        os.makedirs(dataset_folder, exist_ok=True)

        window_size = self.window_length
        overlap = self.window_overlap
        step_size = window_size - overlap

        data_info_list = []

        for i in tqdm(range(0, len(data) - window_size + 1, step_size), desc=f"Windowing data for {patient_id}_{session_id}", leave=False):
            windowed_data = data.iloc[i:i+window_size] if is_csv else data[i:i+window_size]
            if windowed_data.shape[0] < window_size:
                continue

            windowed_data_np = windowed_data.to_numpy() if is_csv else windowed_data.cpu().numpy()
            file_name = f"{patient_id}_session_{session_id}_window_{i}_ws{window_size}_ol{overlap}.csv"
            file_path = os.path.join(dataset_folder, file_name)
            pd.DataFrame(windowed_data_np, columns=data.columns if is_csv else None).to_csv(file_path, index=False)
            data_info_list.append({"file_name": file_name, "file_path": file_path})

        data_info_df = pd.DataFrame(data_info_list)
        data_info_df.to_csv(os.path.join(self.config.dataset_root, dataset_name, f"{split}_info.csv"), index=False, mode='a', header=not os.path.exists(os.path.join(self.config.dataset_root, dataset_name, f"{split}_info.csv")))

if __name__ == "__main__":
    import sys  # you need this to add path to utils folder
    sys.path.append('../utils')
    from configs import config_general

    config = config_general(
        data_folder_name="../../datacollection/vicon",
        dataset_root="../../datasets",
        dataset_name="two_subject_downsampled",
        window_length=100,
        window_overlap=50,
        input_format="csv",
        num_patients=2,
    )
    data_sharder = DataSharder(config)
    data_sharder.load_data()
