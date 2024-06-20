#SHOULD BE GOOD TO GO
import os
import random
import numpy as np
from data_utils import get_data_from_wav_file
from scipy.io import wavfile
import pandas as pd

import torchaudio.transforms as T

import torch
class datasharder_imu_joints: 

    #define config
    def __init__(self,config, sample_rate):
        self.config = config
        self.sample_rate=sample_rate

    #length of data
    def __len__(self):
        return len(self.data)

    #load data
    def load_data(self):        
        
        #where data is located
        data_folder_path = self.config.data_folder_name
        
        #access data folder
        patient_folders_list = [f for f in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, f))]
        
        random.shuffle(patient_folders_list)
        
        # Split the list into training and testing subsets
        training_patients = patient_folders_list[:self.config.num_patients-1]
        testing_patients = patient_folders_list[self.config.num_patients-1:]
                
        #function to process patients in training/testing list
        def process_patients (patient_list_type):
            
            #total data
            totaldata_list = []  # shape = [patientid, sessionsid, emg_channels+joints_channels]
            
            #for patient in data folder
            for patient_id in patient_list_type:
                
                #create a patient data list for each patient
                patient_data_list = []  # shape [sessionsid, emg_channels+joints_channels]
            
                #for the session from the patients data 
                for session_index in range(self.config.num_sessions):
                    
                    #IMU
                    imu_file_path = data_folder_path+"/"+ patient_id+"/"+f"run{session_index}_IMU.wav"
                
                    #JOINT 
                    joints_file_path = data_folder_path+"/"+ patient_id+"/"+f"run{session_index}_JOINTS.wav"
                
                    #grab imu data and sr
                    imu_data, imu_sample_rate = get_data_from_wav_file(imu_file_path)  # [time_steps, channels] ex [1000,8]
                
                    #grab joint data and sr
                    joints_data, joints_sample_rate = get_data_from_wav_file(joints_file_path)  # [time_steps, channels] ex [1000,2]               
                
                    #give status of the loaded imu and joint data
                    print(f"Loaded {imu_file_path}|{joints_file_path}\n__________________________________________")
                    
                    imu_data = torch.tensor(imu_data, dtype=torch.float32)
                    joints_data = torch.tensor(joints_data, dtype=torch.float32)
                
                    if imu_sample_rate != self.sample_rate:
                        imu_data = T.Resample(imu_sample_rate, self.sample_rate)(imu_data.T).T
                    if joints_sample_rate != self.sample_rate:
                        joints_data = T.Resample(joints_sample_rate, self.sample_rate)(joints_data.T).T
                
                    dataset_length = min(imu_data.shape[0], joints_data.shape[0])
                    imu_data = imu_data[:dataset_length]
                    joints_data = joints_data[:dataset_length]
                
                    imu_data = (imu_data - imu_data.mean(dim=0)) / imu_data.std(dim=0)
                    joints_data = (joints_data - joints_data.mean(dim=0)) / joints_data.std(dim=0)      
                
                    #combine imu and joint data
                    combined_data = torch.cat((imu_data, joints_data), dim=1)  # stack [time_steps, emg_channels + joint_channels]
                    
                    #add the combined data to the patient data list
                    patient_data_list.append(combined_data)
            
                #add to the list of total data 
                totaldata_list.append(patient_data_list)
        
            #define total dataset
            result = [torch.stack(patient_data_list) for patient_data_list in totaldata_list]
            
            #return processed list
            return result

        training_patient_list = process_patients(training_patients)
        testing_patient_list = process_patients(testing_patients)
        
        #return training and testing patient data list
        return training_patient_list, testing_patient_list 
    
    #function to window data
    def window_data(self, type_list, split):
        
        #where to send windowed data
        if split == "train":
            dataset_folder = os.path.join(self.config.dataset_root, self.config.dataset_train_name)
        if split == "test":
            dataset_folder = os.path.join(self.config.dataset_root, self.config.dataset_test_name)
            
        #window size
        window_size = self.config.window_length
        
        #where to store all windowed data
        all_data_windowed = []
        
        # To store information about each window
        data_info_list = []  

        #if where to send windowed data doesn't exist
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        
        #n patients
        patient_counter = 0       
        
        #for patient in data from above
        for patient_data in type_list:
            
            #num of sessions
            session_counter = 0
            
            #for each session in patient data
            for session_data in patient_data:
                
                #for windowing size within length of session data
                for i in range(0, len(session_data), window_size):
                    
                    #create windowed data
                    windowed_data = session_data[i:i+window_size]
                    
                    # Skip the last window if it's smaller than the window size
                    if windowed_data.shape[0] < window_size:
                        continue
                    
                    # Convert windowed data to NumPy array if it is a tensor
                    if isinstance(windowed_data, torch.Tensor):
                        windowed_data = windowed_data.cpu().numpy()
                    
                    #set and send windowed data to folder location
                    file_name = f"patient_{patient_counter}_session_{session_counter}_window_{i}.wav"
                    file_path = os.path.join(dataset_folder, file_name)
                    wavfile.write(file_path, self.sample_rate, windowed_data)
                    
                    # Append info to list
                    data_info_list.append({"file_name": file_name, "file_path": file_path})
                    
                    #notify finished windowed data
                    print(f"Saved window to {file_path}")
                
                #session check
                session_counter += 1
            
            #patient check
            patient_counter += 1

        # Convert list to DataFrame
        data_info_df = pd.DataFrame(data_info_list)

        #return windowed data path 
        return data_info_df
    
    #function to save windowed data to config specified destination
    def save_windowed_data(self, type_list, split):

        #start process
        print(f"Saving windowed data to {self.config.dataset_root}")
        
        #window data
        data_info_df = self.window_data(type_list, split)
        
        #training dataset
        if split == "train":
            data_info_df.to_csv(os.path.join(self.config.dataset_root, f"{self.config.dataset_train_name}_info.csv"))
            
        #testing dataset
        if split == "test":
            data_info_df.to_csv(os.path.join(self.config.dataset_root, f"{self.config.dataset_test_name}_info.csv"))
            
        #end process
        print(f"Saved windowed data to {self.config.dataset_root}")