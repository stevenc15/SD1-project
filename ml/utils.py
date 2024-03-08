class config: #hyperparameters
    batch_size = 16
    epochs = 10
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class dataloader: #eclusivly for loading data

    older_name = 'data'
    batch_size = config.batch_size
    num_sessions=12
    num_channels_emg=None
    num_channels_joints=None
    seed = 42
    preprocessing_function = [fft,wavelet,waveletandfft]
    data_unprocesses = np.array([])


    def __init__(self)


    def get_data_from_file(file_name) #load from wav files

    def to_tensor(data) #convert to tensor

    def load_emg_joint_pairs() #from self.folder_name
        totaldata=[] #shape = [patientid,sessionsid,emg_channels+joints_channels]
        for folder in data
            patient_data = [] #shape [sessionsid,emg_channels+joints_channels]
            for session_index in range(self.num_sessions):
                sessionr_data=[]
                if self.num_channels_emg is None:
                    self.num_channels_emg = data.shape[1]
                    self.num_channels_joints = data.shape[1]

                get_data_from_file("run{session_index}_EMG") #[time_steps,channels] ex [1000,8]
                get_data_from_file("run{session_index}_JOINTS") #[time_steps,channels] ex [1000,2]
                
                stack #[time_steps,_emgchannels+joint_channels] ex[1000,10]
            #patients_data s
        
        self.data = np.array(totaldata)

    
                
def save_model(model, name):

def load_model(path_to_model_file):

def fit_model_on_data(model_object, data_loader, config,model_training_runner):
    define optimizer with config
    

def plot_model_history(history):



list_of_models = [DQN,CNN]
f
way to save all models, name models based on config
way to load all models
save all model
