import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

from losses import RMSELoss

def train_kinematics_light(train_loader,val_loader, learn_rate, EPOCHS, model,filename,device):

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


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data_features_1D, data_features_2D, data_acc,data_gyr, data_targets) in enumerate(train_loader):
            optimizer.zero_grad()

            target_output= model(data_acc[:,:,18:24].to(device).float(),data_gyr[:,:,18:24].to(device).float())

            loss_1=criterion(target_output, data_targets.to(device).float())

            loss_1.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_features_1D, data_features_2D,data_acc,data_gyr, data_targets in val_loader:

                output= model(data_acc[:,:,18:24].to(device).float(),data_gyr[:,:,18:24].to(device).float())
                val_loss += criterion(output, data_targets.to(device).float())

        val_loss /= len(val_loader)

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



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model

def train_kinematics(train_loader,val_loader, learn_rate, EPOCHS, model, filename,device):
    if torch.cuda.is_available():
        model.cuda()
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    running_loss = 0
    start_time = time.time()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data_features_1D, data_features_2D, data_acc, data_gyr, data_targets) in enumerate(train_loader):
            optimizer.zero_grad()
            output_1, output_2, output_3, output = model(data_features_1D[:, :, 36:48].to(device).float(),
                                                         data_features_2D.to(device).float())
            regularization_loss = 0.0
            if hasattr(model.output_GRU, 'regularizer_loss'):
                regularization_loss += model.output_GRU.regularizer_loss()
            if hasattr(model.output_C1, 'regularizer_loss'):
                regularization_loss += model.output_C1.regularizer_loss()
            if hasattr(model.output_C2, 'regularizer_loss'):
                regularization_loss += model.output_C2.regularizer_loss()

            loss = criterion(output_1, data_targets.to(device).float()) + criterion(output_2, data_targets.to(device).float()) \
                   + criterion(output_3, data_targets.to(device).float()) + criterion(output, data_targets.to(device).float()) + regularization_loss
            loss_1 = criterion(output, data_targets.to(device).float())

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss = running_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_features_1D, data_features_2D, data_acc, data_gyr, data_targets in val_loader:
                output_1, output_2, output_3, output = model(data_features_1D[:, :, 36:48].to(device).float(),
                                                             data_features_2D.to(device).float())
                val_loss += criterion(output, data_targets.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        print(f"Epoch: {epoch + 1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss = 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Stopping early after {epoch + 1} epochs")
            break

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    return model
