
#IMPORTS

import time #used to measure model 

import torch #for making, training the model and processing the data in pytorch

from loss import RMSELoss

def train_SD(device, alpha, val_loader, train_loader, learn_rate, EPOCHS, student, teacher, filename, k_1s, k_2s, k_3s, k_4s, k_1t, k_2t, k_3t, k_4t):

    if torch.cuda.is_available():
      student.cuda()
    # Defining loss function and optimizer
    criterion =RMSELoss()
    # criterion =correlation_coefficient_loss_joint_pytorch()

    # criterion=PearsonCorrLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=learn_rate)
    # optimizer_1 = torch.optim.Adam(student.model_1.parameters(), lr=learn_rate)
    # optimizer_2 = torch.optim.Adam(student.cnn_1D.parameters(), lr=learn_rate)
    # optimizer_3 = torch.optim.Adam(student.cnn_2D.parameters(), lr=learn_rate)

    # optimizer_t = torch.optim.Adam(teacher.parameters(), lr=learn_rate)

    # optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # student.model_1.train()
        # student.cnn_1D.train()
        # student.cnn_2D.train()
        student.train()

        for i, (data_features_1D, data_features_2D, data_targets) in enumerate(train_loader):



            # with torch.no_grad():
            #  output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())



# ###################################################################################################################################

            # optimizer_1.zero_grad()

            # output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            # loss_1=criterion(output_1, data_targets.to(device).float())+alpha*criterion(output_1, output_1t)

            # loss_1.backward()
            # optimizer_1.step()

# ###################################################################################################################################

            # optimizer_2.zero_grad()

            # output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            # # with torch.no_grad():
            # #  output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())

            # loss_2=criterion(output_2, data_targets.to(device).float())+alpha*criterion(output_2, output_2t)

            # loss_2.backward()
            # optimizer_2.step()

# ###################################################################################################################################


            # optimizer_3.zero_grad()

            # output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            # # with torch.no_grad():
            # #  output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())

            # loss_3=criterion(output_3, data_targets.to(device).float())+alpha*criterion(output_3, output_3t)

            # loss_3.backward()
            # optimizer_3.step()

# ###################################################################################################################################



            optimizer.zero_grad()

            output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())

            with torch.no_grad():
             output_1t, output_2t, output_3t, outputt= teacher(data_features_1D[:,:,k_1t:k_2t].to(device).float(),data_features_2D[:,:,:,k_3t:k_4t].to(device).float())

            loss=criterion(output, data_targets.to(device).float())+alpha*criterion(output, outputt)

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
                output_1, output_2, output_3, output= student(data_features_1D[:,:,k_1s:k_2s].to(device).float(),data_features_2D[:,:,:,k_3s:k_4s].to(device).float())
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
            torch.save(student.state_dict(), filename)
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



    return student