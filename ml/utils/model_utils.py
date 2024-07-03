#SHOULD BE GOOD TO GO
import torch
from matplotlib import pyplot as plt
import numpy as np

# Plotting function
def plot_data_kinelight(x, y, config, scaler, title_prefix):
    imu_channels = config.channels_imu
    joint_channels = config.channels_joints

    num_imu_channels = len(imu_channels)
    num_joint_channels = len(joint_channels)

    fig, axs = plt.subplots(num_imu_channels + num_joint_channels, 2, figsize=(15, (num_imu_channels + num_joint_channels) * 5))
    
    # Plot normalized IMU data
    for i, channel in enumerate(imu_channels):
        axs[i, 0].plot(x[0, :, i].cpu().numpy())
        axs[i, 0].set_title(f"{title_prefix} - Normalized IMU Channel: {channel}")
        axs[i, 0].set_xlabel("Time")
        axs[i, 0].set_ylabel("Amplitude")

    # Plot normalized Joint data
    for i, channel in enumerate(joint_channels):
        axs[i + num_imu_channels, 0].plot(y[0, :, i].cpu().numpy())
        axs[i + num_imu_channels, 0].set_title(f"{title_prefix} - Normalized Joint Channel: {channel}")
        axs[i + num_imu_channels, 0].set_xlabel("Time")
        axs[i + num_imu_channels, 0].set_ylabel("Amplitude")

    # Concatenate IMU and Joint data for inverse scaling
    combined_data_normalized = np.concatenate([x[0].cpu().numpy(), y[0].cpu().numpy()], axis=1)
    combined_data_unnormalized = scaler.inverse_transform(combined_data_normalized)

    # Split the unnormalized data back into IMU and Joint
    imu_data_unnormalized = combined_data_unnormalized[:, :num_imu_channels]
    joint_data_unnormalized = combined_data_unnormalized[:, num_imu_channels:]

    # Plot unnormalized IMU data
    for i, channel in enumerate(imu_channels):
        axs[i, 1].plot(imu_data_unnormalized[:, i])
        axs[i, 1].set_title(f"{title_prefix} - Unnormalized IMU Channel: {channel}")
        axs[i, 1].set_xlabel("Time")
        axs[i, 1].set_ylabel("Amplitude")

    # Plot unnormalized Joint data
    for i, channel in enumerate(joint_channels):
        axs[i + num_imu_channels, 1].plot(joint_data_unnormalized[:, i])
        axs[i + num_imu_channels, 1].set_title(f"{title_prefix} - Unnormalized Joint Channel: {channel}")
        axs[i + num_imu_channels, 1].set_xlabel("Time")
        axs[i + num_imu_channels, 1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

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
