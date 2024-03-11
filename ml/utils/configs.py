import torch

class config_general: #hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self,**kwargs):
        self.batch_size = kwargs.get('batch_size',16)
        self.epochs = kwargs.get('epochs',10)
        self.lr = kwargs.get('lr',0.001)
        self.scheduler = kwargs.get('scheduler',None)
        self.num_channels_emg = kwargs.get('num_channels_emg',None)
        self.num_channels_joints = kwargs.get('num_channels_joints',None)
        self.num_sessions = kwargs.get('num_sessions',None)
        self.seed = kwargs.get('seed',42)
        self.data_folder_name = kwargs.get('data_folder_name','default_data_folder_name')
        self.dataset_root = kwargs.get('dataset_root','default_dataset_root')
        self.dataset_name = kwargs.get('dataset_name','default_dataset_name')
        self.train_test_split = kwargs.get('train_test_split',0.8)
        self.window_length = kwargs.get('window_length',100) #need to look up best value later
        self.emg_transforms = kwargs.get('emg_transforms',[])
        self.joint_transforms = kwargs.get('joint_transforms',[])

        hyperparameters = [
        f"dataset_name_{self.dataset_name}",
        f"batch_size_{self.batch_size}",
        f"epochs_{self.epochs}",
        f"lr_{self.lr}",
        f"num_channels_emg_{self.num_channels_emg}",
        f"num_channels_joints_{self.num_channels_joints}",
        f"num_sessions_{self.num_sessions}",
        f"seed_{self.seed}",
        f"data_folder_name_{self.data_folder_name}",
        f"train_test_split_{self.train_test_split}",
        f"window_length_{self.window_length}"
        ]

        self.hyperparameters_str = "_".join(hyperparameters)




