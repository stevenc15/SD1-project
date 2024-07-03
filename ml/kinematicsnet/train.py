import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from losses import RMSELoss
import numpy as np

"""## Lightweight Model

### Training Function
"""

def train_kinematics_light(train_loader,val_loader,config, learn_rate, EPOCHS, model,filename,device):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=learn_rate)

    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        
        for i, (data_inputs, data_targets) in enumerate(train_loader):
            optimizer.zero_grad()

            target_output= model(data_inputs[:,:,:len(config.channels_imu) // 2].to(device).float(),data_inputs[:,:,len(config.channels_imu) // 2:].to(device).float())

            loss_1=criterion(target_output, data_targets.to(device).float())

            loss_1.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)
        train_losses.append(train_loss)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_inputs, data_targets in val_loader:

                output= model(data_inputs[:,:,:len(config.channels_imu) // 2].to(device).float(),data_inputs[:,:,len(config.channels_imu) // 2:].to(device).float())
                val_loss += criterion(output, data_targets.to(device).float()).item()


        val_loss /= len(val_loader)
        val_losses.append(val_loss)


        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    # After training loop and early stopping
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")

    # Convert losses to numpy arrays for plotting
    train_losses_np = np.array(train_losses)
    val_losses_np = np.array(val_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_np, label='Training Loss')
    plt.plot(val_losses_np, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
