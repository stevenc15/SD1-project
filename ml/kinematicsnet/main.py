import torch
from train import train_kinematics
from models import Gait_Net
from data_preparation import train_loader, val_loader, test_loader

# Set learning rate
lr = 0.001

# Instantiate the model
model = Gait_Net(12, 2, 100)

# Train the model
path = 'path/to/save/model/'
gait_Net = train_kinematics(train_loader, lr, 30, model, path + 'gait_net_kinematics.pth')

# Load and evaluate the model
gait_Net = Gait_Net(12, 2, 100)
gait_Net.load_state_dict(torch.load(path + 'gait_net_kinematics.pth'))
gait_Net.to(device)
gait_Net.eval()



# Iterate through batches of test data
with torch.no_grad():
    for i, (data_features_1D, data_features_2D, data_acc, data_gyr, data_targets) in enumerate(test_loader):
        output_1, output_2, output_3, output = gait_Net(data_features_1D[:, :, 36:48].to(device).float(),
                                                        data_features_2D.to(device).float())
        if i == 0:
            yhat_5 = output
            test_target = data_targets
        else:
            yhat_5 = torch.cat((yhat_5, output), dim=0)
            test_target = torch.cat((test_target, data_targets), dim=0)

yhat_4 = yhat_5.detach().cpu().numpy()
test_y = test_target.detach().cpu().numpy()
print(yhat_4.shape)

# Assuming w is defined somewhere in the script
w = 100
yhat_5 = yhat_4.reshape((yhat_4.shape[0] * w, 6))
test_y_r = test_y.reshape((test_y.shape[0] * w, 6))

print(yhat_4.shape)

# Assuming unpack_dataset_present and prediction_test are defined elsewhere
yhat_up = unpack_dataset_present(np.array(yhat_5))
test_y_up = unpack_dataset_present(np.array(test_y_r))

print(yhat_up.shape, test_y_up.shape)

yhat_up = yhat_up.reshape(int(len(yhat_up) / 6), 6)
test_y_up = test_y_up.reshape(int(len(test_y_up) / 6), 6)

print(yhat_up.shape, test_y_up.shape)

rmse, p = prediction_test(np.array(yhat_up), np.array(test_y_up))

print(rmse[0])
print(rmse[1])
print(rmse[2])

m = np.mean(rmse)

print('\n')
print(m)
print('\n')
print(p[0])
print(p[1])
print(p[2])
print('\n')
print(np.mean(p))
