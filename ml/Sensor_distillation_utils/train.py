
#IMPORTS
import torch #for making, training the model and processing the data in pytorch
import time
from loss import RMSELoss

def train_kinematics(train_loader, val_loader, device,  learn_rate, EPOCHS, model,filename,k_1,k_2,k_3,k_4):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    criterion = RMSELoss()
    # criterion =correlation_coefficient_loss_joint_pytorch()

    # criterion=PearsonCorrLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    optimizer_1 = torch.optim.Adam(model.model_1.parameters(), lr=learn_rate)
    optimizer_2 = torch.optim.Adam(model.cnn_1D.parameters(), lr=learn_rate)
    optimizer_3 = torch.optim.Adam(model.cnn_2D.parameters(), lr=learn_rate)

    optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # model.model_1.train()
        # model.cnn_1D.train()
        # model.cnn_2D.train()
        model.train()

        for i, (data_features_1D, data_features_2D, data_targets) in enumerate(train_loader):


# ###################################################################################################################################

            # optimizer_1.zero_grad()

            # output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            # loss_1=criterion(output_1, data_targets.to(device).float())

            # loss_1.backward()
            # optimizer_1.step()

# ###################################################################################################################################

            # optimizer_2.zero_grad()

            # output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            # loss_2=criterion(output_2, data_targets.to(device).float())

            # loss_2.backward()
            # optimizer_2.step()

# ###################################################################################################################################


            # optimizer_3.zero_grad()

            # output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            # loss_3=criterion(output_3, data_targets.to(device).float())

            # loss_3.backward()
            # optimizer_3.step()

# ###################################################################################################################################



            optimizer.zero_grad()

            output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
            loss=criterion(output, data_targets.to(device).float())

            loss.backward()
            optimizer.step()


###################################################################################################################################




            # Compute the regularization loss for the custom linear layers
            # regularization_loss = 0.0
            # if hasattr(model.output_GRU, 'regularizer_loss'):
            #     regularization_loss += model.output_GRU.regularizer_loss()
            # if hasattr(model.output_C1, 'regularizer_loss'):
            #     regularization_loss += model.output_C1.regularizer_loss()
            # if hasattr(model.output_C2, 'regularizer_loss'):
            #     regularization_loss += model.output_C2.regularizer_loss()

            # loss=criterion(output_1, data_targets.to(device).float())+criterion(output_2, data_targets.to(device).float())\
            # +criterion(output_3, data_targets.to(device).float())+criterion(output, data_targets.to(device).float())

            loss_1=criterion(output, data_targets.to(device).float())

            # loss.backward()
            # optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_features_1D, data_features_2D, data_targets in val_loader:
                output_1, output_2, output_3, output= model(data_features_1D[:,:,k_1:k_2].to(device).float(),data_features_2D[:,:,:,k_3:k_4].to(device).float())
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