import torch

class config_general: #hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self,**kwargs):
        self.batch_size = kwargs.get('batch_size',16) #check
        self.epochs = kwargs.get('epochs',10) #check
        self.lr = kwargs.get('lr',0.001) #check
        self.scheduler = kwargs.get('scheduler',None) #check
        self.num_channels_imu = kwargs.get('num_channels_imu',None) #check
        self.num_channels_joints = kwargs.get('num_channels_joints',None) #check
        self.num_sessions = kwargs.get('num_sessions',None) #check
        self.num_patients = kwargs.get('num_patients',None) #check
        self.seed = kwargs.get('seed',42) #check
        self.data_folder_name = kwargs.get('data_folder_name','default_data_folder_name') #check
        self.dataset_root = kwargs.get('dataset_root','default_dataset_root') #check
        self.dataset_train_name = kwargs.get('dataset_train_name','default_dataset_train_name') #check
        self.dataset_test_name = kwargs.get('dataset_test_name','default_dataset_test_name') #check
        self.window_length = kwargs.get('window_length',100) #need to look up best value later #check
        self.imu_transforms = kwargs.get('imu_transforms',[]) #check
        self.joint_transforms = kwargs.get('joint_transforms',[]) #check
        self.hidden_size = kwargs.get('hidden_size', 256) #check
        self.num_layers = kwargs.get('num_layers', 6)
        self.input_size = kwargs.get('input_size', 3)
        self.output_size = kwargs.get('output_size', 3)
        
        hyperparameters = [
        f"batch_size_{self.batch_size}",
        f"epochs_{self.epochs}",
        f"lr_{self.lr}",
        f"num_channels_imu_{self.num_channels_imu}",
        f"num_channels_joints_{self.num_channels_joints}",
        f"num_sessions_{self.num_sessions}",
        f"num_patients_{self.num_patients}",
        f"seed_{self.seed}",
        f"data_folder_name_{self.data_folder_name}",
        f"dataset_root_{self.dataset_root}",
        f"dataset_train_name_{self.dataset_train_name}",
        f"dataset_test_name_{self.dataset_test_name}", 
        f"window_length_{self.window_length}",
        f"imu_transforms_{self.imu_transforms}",
        f"joints_transforms_{self.joints_transforms}",
        f"hidden_size_{self.hidden_size}",
        f"num_layers_{self.num_layers}",
        f"input_size_{self.input_size}",
        f"output_size_{self.output_size}"
        ]

        self.hyperparameters_str = "_".join(hyperparameters)




