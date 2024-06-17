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
        
        #total data
        totaldata_list = []  # shape = [patientid, sessionsid, emg_channels+joints_channels]
        
        #where data is located
        data_folder_path = self.config.data_folder_name
        
        #access data folder
        patient_folders_list = [f for f in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, f))]
        
        random.shuffle(patient_folders_list)
        
        # Split the list into training and testing subsets
        training_patients = patient_folders_list[:self.config.num_patients-1]
        testing_patients = patient_folders_list[self.config.num_patients-1:]
        
        training_patient_list = process_patients(training_patients)
        testing_patient_list = process_patients(testing_patients)
        
        def process_patients (patient_list_type):
            
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
                
                    imu_data = torch.tensor(imu_data, dtype=torch.float32)
                    joints_data = torch.tensor(joints_data, dtype=torch.float32)
                
                    if imu_sample_rate != self.sample_rate:
                        imu_data = T.Resample(imu_sample_rate, self.sample_rate)(imu_data.T).T
                    if joints_sample_rate != self.sample_rate:
                        joints_data = T.Resample(joints_sample_rate, self.sample_rate)(joints_data.T).T
                
                    dataset_length = min(len(self.input_wav), len(self.target_wav))
                
                    imu_data = imu_data[:dataset_length]
                    joints_data = joints_data[:dataset_length]
                
                    imu_data = (imu_data - imu_data.mean()) / imu_data.std()
                    joints_data = (joints_data - joints_data.mean()) / joints_data.std()
                
                    #give status of the loaded imu and joint data
                    print(f"Loaded {imu_file_path}|{joints_file_path}\n__________________________________________")
                
                    #combine imu and joint data into one hstack
                    combined_data = np.hstack((imu_data, joints_data))  # stack [time_steps, emg_channels+joint_channels] ex[1000,10]
                
                    #add the combined data to the patient data list
                    patient_data_list.append(combined_data)
            
                #add to the list of total data
                totaldata_list.append(patient_data_list)
        
            #define total dataset
            result = np.array(totaldata_list)
            
            return result

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
    
    #function to save windowed data to config specified destination
    def save_windowed_data(self, type_list, split):

        print(f"Saving windowed data to {self.config.dataset_root}")
        data_info_df = self.window_data(type_list, split)
        if split == "train":
            data_info_df.to_csv(os.path.join(self.config.dataset_root, f"{self.config.dataset_train_name}_info.csv"))
        if split == "test":
            data_info_df.to_csv(os.path.join(self.config.dataset_root, f"{self.config.dataset_test_name}_info.csv"))
        print(f"Saved windowed data to {self.config.dataset_root}")

#////////////////////////////////////////////////////////////////////////////////////////////////////
        #path not found to wav files #DATASHARDERS
        #if not os.path.exists(input_wav) or not os.path.exists(target_wav): 
            #raise FileNotFoundError(f"Input file {input_wav} or target file {target_wav} not found.")

        #path found #DATASHARDERS
        #try:
            
            #read in sample rate and data from wav files #DATASHARDERS
            #self.input_sr, imu_data = wavfile.read(input_wav)
            #self.target_sr, joints_data = wavfile.read(target_wav)
        
        #error in loading data from files #DATASHARDERS
        #except Exception as e:
            #raise RuntimeError(f"Error loading wav files: {e}")

        #make data into tensor shape #DATASHARDERS
        #imu_data = torch.tensor(self.input_wav, dtype_list=torch.float32)
        #joints_data = torch.tensor(self.target_wav, dtype_list=torch.float32)
      
        #adjust sample rate of input and target to desired sr #DATASHARDER
        #if self.input_sr != sample_rate:
            #self.input_wav = T.Resample(self.input_sr, sample_rate)(self.input_wav.T).T
        #if self.target_sr != sample_rate:
            #self.target_wav = T.Resample(self.target_sr, sample_rate)(self.target_wav.T).T

        #adjust length of Dataset to be the smallest size between the target and input data so #DATASHARDERS
        #that both datasets are of equal length
        #self.length = min(len(self.input_wav), len(self.target_wav))
        #adjust both datasets to include the first self.length's data
        #self.input_wav = self.input_wav[:self.length]
        #self.target_wav = self.target_wav[:self.length]
 
        # Normalize data, subtract the mean from each data point and divide it by the standard deviation #DATASHARDERS
        #self.input_wav = (self.input_wav - self.input_wav.mean()) / self.input_wav.std()
        #self.target_wav = (self.target_wav - self.target_wav.mean()) / self.target_wav.std()