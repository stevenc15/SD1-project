import os
import numpy as np
from data_utils import get_data_from_wav_file
from scipy.io import wavfile
import pandas as pd

class dataloader_emg_joints: 

    def __init__(self,config):
        self.config = config

    def __len__(self):
        return len(self.data)

    def load_data(self):
        totaldata_list = []  # shape = [patientid, sessionsid, emg_channels+joints_channels]
        data_folder_path = self.config.data_folder_name
        patient_folders_list = [f for f in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, f))]
        
        for patient_id in patient_folders_list:
            patient_data_list = []  # shape [sessionsid, emg_channels+joints_channels]
            for session_index in range(self.config.num_sessions):
                emg_file_path = data_folder_path+"/"+ patient_id+"/"+f"run{session_index}_EMG.wav"
                joints_file_path = data_folder_path+"/"+ patient_id+"/"+f"run{session_index}_JOINTS.wav"
                emg_data,self.sample_rate = get_data_from_wav_file(emg_file_path)  # [time_steps, channels] ex [1000,8]
                joints_data,_ = get_data_from_wav_file(joints_file_path)  # [time_steps, channels] ex [1000,2]
                print(f"Loaded {emg_file_path}|{joints_file_path}\n__________________________________________")
                combined_data = np.hstack((emg_data, joints_data))  # stack [time_steps, emg_channels+joint_channels] ex[1000,10]
                patient_data_list.append(combined_data)
            
            totaldata_list.append(patient_data_list)
        
        self.data = np.array(totaldata_list)

    def window_data(self, dataset_folder):
        window_size = self.config.window_length
        all_data_windowed = []
        data_info_list = []  # To store information about each window


        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        
        patient_counter = 0
        for patient_data in self.data:
            session_counter = 0
            for session_data in patient_data:
                for i in range(0, len(session_data), window_size):
                    windowed_data = session_data[i:i+window_size]
                    if windowed_data.shape[0] < window_size:
                        # Skip the last window if it's smaller than the window size
                        continue
                    
                    file_name = f"patient_{patient_counter}_session_{session_counter}_window_{i}.wav"
                    file_path = os.path.join(dataset_folder, file_name)
                    wavfile.write(file_path, self.sample_rate, windowed_data)
                    
                    # Append info to list
                    data_info_list.append({"file_name": file_name, "file_path": file_path})
                    
                    print(f"Saved window to {file_path}")
                session_counter += 1
            patient_counter += 1

        # Convert list to DataFrame
        data_info_df = pd.DataFrame(data_info_list)

        return data_info_df
    
    def save_windowed_data(self,dataset_path,dataset_name):
        print(f"Saving windowed data to {dataset_path}")
        data_info_df = self.window_data(os.path.join(dataset_path, dataset_name))
        data_info_df.to_csv(os.path.join(dataset_path, f"{dataset_name}_info.csv"))
        print(f"Saved windowed data to {dataset_path}")


    def load_emg_joint_pairs(self):
        print("Loading emg_joint_pairs")
        self.load_data()
        print("Loaded emg_joint_pairs")
        print(f"Data shape: {self.data.shape}")