import torch
from matplotlib import pyplot as plt

def save_model(model,history, name):
    torch.save(model.state_dict(), name)
    torch.save(history, name+"_history")
    print(f"{model.__str__} saved as {name}")

def load_model(path_to_model_file,model):
    model.load_state_dict(torch.load(path_to_model_file))
    history = torch.load(path_to_model_file+"_history")
    print(f"{model.__str__} loaded from {path_to_model_file}")

    return model, history


def plot_model_history(history, model_name):
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['test_loss'], label='test loss')
    plt.legend()
    plt.title(f"{model_name} loss")
    plt.show()