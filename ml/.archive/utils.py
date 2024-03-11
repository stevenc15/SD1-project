import numpy as np
import torch
from scipy.io import wavfile
import os
class config_general: #hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self,**kwargs):
        self.train_test_split = kwargs.get('train_test_split',default=0.8)
        self.batch_size = kwargs.get('batch_size',default=16)
        self.epochs = kwargs.get('epochs',default=10)
        self.lr = kwargs.get('lr',default=0.001)
        self.scheduler = kwargs.get('scheduler',default=None)
        self.num_channels_emg = kwargs.get('num_channels_emg',default=None)
        self.num_channels_joints = kwargs.get('num_channels_joints',default=None)
        self.num_sessions = kwargs.get('num_sessions',default=None)
        self.seed = kwargs.get('seed',default=42)
        self.data_folder_name = kwargs.get('data_folder_name',default='default_data_folder_name')


        hyperparameters = [
        f"batch_size_{self.batch_size}",
        f"epochs_{self.epochs}",
        f"lr_{self.lr}",
        f"num_channels_emg_{self.num_channels_emg}",
        f"num_channels_joints_{self.num_channels_joints}",
        f"num_sessions_{self.num_sessions}",
        f"seed_{self.seed}",
        ]

        self.hyperparameters_str = "_".join(hyperparameters)



class dataloader_emg_joints: #eclusivly for loading data

    train_data = None
    test_data = None

    def __init__(self,config):
        self.config = config

    def get_data_from_wav_file(self,filename) #return data shape [time_steps,channels]
        sample_rate, data = wavfile.read(filename)
        print(f"Sample rate: {sample_rate}Hz")
        print(f"Data shape: {data.shape}")

        return data, sample_rate

    def to_tensor(data) #convert to tensor
        pass

    def load_emg_joint_pairs(self):
        totaldata = []  # shape = [patientid, sessionsid, emg_channels+joints_channels]
        data_folder_path = self.config.data_folder_name
        patient_folders = [f for f in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, f))]
        
        for patient_id in patient_folders:
            patient_data = []  # shape [sessionsid, emg_channels+joints_channels]
            for session_index in range(self.num_sessions):
                session_data = []
                emg_file_path = os.path.join(data_folder_path, patient_id, f"run{session_index}_EMG.wav")
                joints_file_path = os.path.join(data_folder_path, patient_id, f"run{session_index}_JOINTS.wav")
                
                emg_data = self.get_data_from_wav_file(emg_file_path)  # [time_steps, channels] ex [1000,8]
                joints_data = self.get_data_from_wav_file(joints_file_path)  # [time_steps, channels] ex [1000,2]
                
                # Assuming the time steps are aligned between EMG and joints data
                combined_data = np.hstack((emg_data, joints_data))  # stack [time_steps, emg_channels+joint_channels] ex[1000,10]
                patient_data.append(combined_data)
            
            totaldata.append(patient_data)
        
        self.data = np.array(totaldata)

    
                
def save_model(model, name):
    torch.save(model.state_dict(), name)
    print(f"{model.__str__} saved as {name}")

def load_model(path_to_model_file,model):
    model.load_state_dict(torch.load(path_to_model_file))
    print(f"{model.__str__} loaded from {path_to_model_file}")

# def fit_model_on_data(model_object, data_loader, config,model_training_runner):
#     define optimizer with config
    

# def plot_model_history(history):

