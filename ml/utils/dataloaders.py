import os
import numpy as np
from data_utils import get_data_from_wav_file

class dataloader_emg_joints: 

    def __init__(self,config):
        self.config = config

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pass
        
    def apply_transforms(self):
        pass

    def load_data(self):
        totaldata_list = []  # shape = [patientid, sessionsid, emg_channels+joints_channels]
        data_folder_path = self.config.data_folder_name
        patient_folders_list = [f for f in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, f))]
        
        for patient_id in patient_folders_list:
            patient_data_list = []  # shape [sessionsid, emg_channels+joints_channels]
            for session_index in range(self.config.num_sessions):
                emg_file_path = data_folder_path+"/"+ patient_id+"/"+f"run{session_index}_EMG.wav"
                joints_file_path = data_folder_path+"/"+ patient_id+"/"+f"run{session_index}_JOINTS.wav"
                emg_data,_ = get_data_from_wav_file(emg_file_path)  # [time_steps, channels] ex [1000,8]
                joints_data,_ = get_data_from_wav_file(joints_file_path)  # [time_steps, channels] ex [1000,2]
                print(f"Loaded {emg_file_path}|{joints_file_path}\n__________________________________________")
                combined_data = np.hstack((emg_data, joints_data))  # stack [time_steps, emg_channels+joint_channels] ex[1000,10]
                patient_data_list.append(combined_data)
            
            totaldata_list.append(patient_data_list)
        
        self.data = np.array(totaldata_list)

    def window_data(self):
        window_size = self.config.window_length
        all_data_windowed = []
        for patient_data in self.data:
            patient_data_windowed = []
            for session_data in patient_data:
                session_data_windowed = []
                for i in range(0, len(session_data), window_size):
                    windowed_data = session_data[i:i+window_size]
                    print(f"Windowed data shape: {windowed_data.shape}")
                    session_data_windowed.append(windowed_data)
                patient_data_windowed.append(session_data_windowed)
            all_data_windowed.append(patient_data_windowed)
                    
        self.data_windowed = np.array(all_data_windowed)
        return self.data_windowed



    def load_emg_joint_pairs(self):
        print("Loading emg_joint_pairs")
        self.load_data()
        print("Loaded emg_joint_pairs")
        print(f"Data shape: {self.data.shape}")