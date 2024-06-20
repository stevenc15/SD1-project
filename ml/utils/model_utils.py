#SHOULD BE GOOD TO GO
import torch
from matplotlib import pyplot as plt
import numpy as np

#function to plot predictions MODEL
def plot_predictions(inputs, targets, predictions, num_channels=3):
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 8))

    for i in range(num_channels):
        axs[i].plot(targets[:, i], label='Ground Truth')
        axs[i].plot(predictions[:, i], label='Prediction')
        axs[i].set_title(f'Target vs Prediction Channel {i + 1}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

#train function for model #MODEL
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)  # Ensure 3D tensor
            targets = targets.to(device)  # Ensure 3D tensor
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)  # Ensure 3D tensor
                targets = targets.to(device)  # Ensure 3D tensor

                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses

#function to plot loss #MODEL
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
#function to evaluate model performance
def evaluate_model(model, data_loader, device):
    model.eval()
    all_inputs, all_targets, all_predictions = [], [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)  # Ensure 3D tensor
            targets = targets.to(device)  # Ensure 3D tensor

            outputs = model(inputs)
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    all_inputs = np.concatenate(all_inputs)
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    return all_inputs, all_targets, all_predictions
