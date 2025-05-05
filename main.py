# ### SHRED for ROMs

import torch
import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

# Local files
import models
from processdata import load_data
from processdata import TimeSeriesDataset

# Directories
top_dir = Path(__file__).parent
data_dir = top_dir / 'datasets'
plasma_dir = data_dir / 'plasma'
fig_dir = top_dir / 'figures'

# We first randomly select 3 sensor locations and set the trajectory length (lags) to 52, which is hyperparameter tuned.

#Load data
v_total = np.load(plasma_dir / 'v_total.npy') #Data for v location
s_total = np.load(plasma_dir / 's_total.npy') #Data for s location
u_total = np.load(plasma_dir / 'u_total.npy') #Data for u location

plasma_data = sio.loadmat(plasma_dir / 'ne.mat') #Load file from matlab
utemp = plasma_data['Data'] # Access plasma data using the data key
X = utemp - np.mean(utemp, axis=0) # Find X by subtracting the data by the mean of their columns (normalization)
Xnorm =  np.max(np.abs(X)) # Returns the absolute value of each element in X and the maximum value in the data
X = X/Xnorm # Normalization technique
#Z score normalization

# Observe shape of data
n2 = (X).shape[0]
m2 = s_total.shape[1]  # svd modes used

num_sensors = 3 # Determines the number of sensors
lags = 52 # Time delay of 52

nx = 257
ny = 256

load_X = v_total.T # Transpose v_total 

sensor_locations_ne = np.random.choice(n2, size=num_sensors, replace=False)
sensor_locations = [0, 1, 2]

load_X = np.hstack((X[sensor_locations_ne,:].T,load_X))

n = (load_X).shape[0]
m = (load_X).shape[1]

mask = np.zeros(n2)           
mask[sensor_locations_ne[0]]=1
mask[sensor_locations_ne[1]]=1
mask[sensor_locations_ne[2]]=1

mask2 = mask.reshape((nx,ny)) 

fig = plt.figure(figsize=(25, 20))
ax = fig.add_subplot(2, 1, 1)
plt.imshow(mask2, cmap='gray')

plt.savefig('measure.pdf')

# We now select indices to divide the data into training, validation, and test sets.

# RECONSTRUCTION MODE
train_indices = np.random.choice(n - lags, size=500, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]

# FORECASTING MODE

scaler = MinMaxScaler()
scaler = scaler.fit(load_X[train_indices])
transformed_X = scaler.transform(load_X)

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

# We train the model using the training and validation datasets.

# Finally, we generate reconstructions from the test set and print mean square error compared to the ground truth.

UTransformer = models.TimeSeries_UTransformer(d_model=128, nhead=16, sequence_length=500, dropout=0.1).to(device)

#validation_errors = models.fit(UTransformer, train_dataset, valid_dataset, batch_size=25, num_epochs=8, lr=0.001, verbose=True, patience=5)
#UTransformer = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(UTransformer, train_dataset, valid_dataset, batch_size=64, num_epochs=3000, lr=1e-3, verbose=True, patience=5)
print(list(validation_errors))

def safe_transform(scaler, data, inverse=False, refit=False):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    original_shape = data.shape
    
    # Reshape to 2D, preserving the first dimension
    reshaped_data = data.reshape(original_shape[0], -1)
    
    if not hasattr(scaler, 'n_features_in_') or refit:
        print(f"Fitting scaler on data with shape {reshaped_data.shape}")
        scaler.fit(reshaped_data)
    elif scaler.n_features_in_ != reshaped_data.shape[1]:
        print(f"Warning: Scaler expects {scaler.n_features_in_} features, but data has {reshaped_data.shape[1]} features.")
        print(f"Refitting scaler on data with shape {reshaped_data.shape}")
        scaler.fit(reshaped_data)
    
    if inverse:
        transformed = scaler.inverse_transform(reshaped_data)
    else:
        transformed = scaler.transform(reshaped_data)
    
    return transformed.reshape(original_shape)

# Initialize separate scalers for X and Y
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

# Fit scalers on training data
scaler_X.fit(train_dataset.X.detach().cpu().numpy().reshape(-1, train_dataset.X.shape[1]))
scaler_Y.fit(train_dataset.Y.detach().cpu().numpy().reshape(-1, train_dataset.Y.shape[1]))

# For input data
X_data = safe_transform(scaler_X, test_dataset.X)
print("X_data shape after transform:", X_data.shape)

# Apply UTransformer
transformed_data = UTransformer(torch.tensor(X_data, device=test_dataset.X.device))
print("Transformed data shape:", transformed_data.shape)

# For test reconstructions
test_recons = safe_transform(scaler_X, transformed_data, inverse=True)
print("Test reconstruction shape:", test_recons.shape)

# For ground truth
test_ground_truth = safe_transform(scaler_Y, test_dataset.Y, inverse=True)
print("Ground truth shape:", test_ground_truth.shape)

# Reshape test_recons to match ground truth
test_recons_reshaped = test_recons.reshape(test_recons.shape[0], -1)[:, :test_ground_truth.shape[1]]
print("Reshaped test reconstruction shape:", test_recons_reshaped.shape)

# Calculate and print the reconstruction error using MSE
mse_reconstruction_error = np.mean((test_recons_reshaped - test_ground_truth) ** 2)
print(f"Reconstruction Mean Squared Error: {mse_reconstruction_error}")

# Print shapes for debugging                             
print("test_recons shape:", test_recons.shape)           
print("test_ground_truth shape:", test_ground_truth.shape)



fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 3, 1)
plt.plot(test_recons_reshaped[10])
ax = fig.add_subplot(2, 3, 4)
plt.plot(test_ground_truth[10])
ax = fig.add_subplot(2, 3, 2)
plt.plot(test_recons_reshaped[50])
ax = fig.add_subplot(2, 3, 5)
plt.plot(test_ground_truth[50])
ax = fig.add_subplot(2, 3, 3)
plt.plot(test_recons_reshaped[150])
ax = fig.add_subplot(2, 3, 6)
plt.plot(test_ground_truth[150])


fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 1, 1)
plt.imshow(test_recons_reshaped[:,3:])
ax = fig.add_subplot(2, 1, 2)
plt.imshow(test_ground_truth[:,3:])



print(test_ground_truth.shape)
print(test_recons_reshaped.shape)



fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 1, 1)
x=test_recons_reshaped[:,[40]]
plt.plot(x)
ax = fig.add_subplot(2, 1, 2)
x=test_ground_truth[:,[40]]
plt.plot(x)



fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 1, 1)
x=test_recons_reshaped[:,[140]]
plt.plot(x)
ax = fig.add_subplot(2, 1, 2)
x=test_ground_truth[:,[140]]
plt.plot(x)



fig = plt.figure(figsize=(15, 20))
mpoint = 20000

for jj in range(14):
    ax = fig.add_subplot(7, 2, jj+1)
    upca = u_total[:, jj*m2:(jj+1)*m2]
    spca = s_total[jj, :]
    vpca1 = test_ground_truth[:, jj*m2+3:(jj+1)*m2+3]
    vpca2 = test_recons[:,jj*m2+3:(jj+1)*m2+3]
    
    u1svd = upca @ np.diag(spca) @ vpca1.T
    u2svd = upca @ np.diag(spca) @ vpca2.T
    
    plt.plot(u1svd[mpoint,100:400], color='gray')
    plt.plot(u2svd[mpoint,100:400])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.axis('off')

   # ax.set_title(f"Plot {jj+1}")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.savefig('timeseries.pdf')
plt.show()


fig = plt.figure(figsize=(20, 20))

loop1=[0,1,2,3,4,5,6]
loop2=[7,8,9,10,11,12,13]

for jj in loop1:
    ax = fig.add_subplot(1, 7, jj+1)

    upca = u_total[:, jj*m2:(jj+1)*m2]
    spca = s_total[jj, :]
    vpca1 = test_ground_truth[:, jj*m2+3:(jj+1)*m2+3]

    u1svd = upca @ np.diag(spca) @ vpca1.T
    snap_true = u1svd[0:nx*ny, jj].reshape((nx, ny)).T
    ax.imshow(snap_true,cmap='RdBu_r', interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()

plt.savefig('comp1.pdf')
plt.show()    

fig = plt.figure(figsize=(20, 20))
for jj in loop1:
    ax = fig.add_subplot(1, 7, jj+1)
    upca = u_total[:, jj*m2:(jj+1)*m2]
    spca = s_total[jj, :]
    #vpca1 = test_ground_truth[:, jj+3:jj+m2+3]
    vpca2 = test_recons_reshaped[:,jj*m2+3:(jj+1)*m2+3]

    #u1svd = upca @ np.diag(spca) @ vpca1.T
    u2svd = upca @ np.diag(spca) @ vpca2.T
    
    #snap_true = u1svd[0:nx*ny, j].reshape((nx, ny))
    snap_test = u2svd[0:nx*ny,jj].reshape((nx,ny)).T
    ax.imshow(snap_test,cmap='RdBu_r', interpolation='bilinear')

    ax.axis('off')
    plt.tight_layout()

plt.savefig('comp2.pdf')
plt.show()    




fig = plt.figure(figsize=(20, 20))
for jj in loop2:
    ax = fig.add_subplot(1, 7, jj-6)

    upca = u_total[:, jj*m2:(jj+1)*m2]
    spca = s_total[jj, :]
    vpca1 = test_ground_truth[:, jj*m2+3:(jj+1)*m2+3]

    u1svd = upca @ np.diag(spca) @ vpca1.T
    snap_true = u1svd[0:nx*ny, jj].reshape((nx, ny)).T
    ax.imshow(snap_true,cmap='RdBu_r', interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()

plt.savefig('comp3.pdf')
plt.show()    

fig = plt.figure(figsize=(20, 20))
for jj in loop2:
    ax = fig.add_subplot(1, 7, jj-6)
    upca = u_total[:, jj*m2:(jj+1)*m2]
    spca = s_total[jj, :]
    #vpca1 = test_ground_truth[:, jj+3:jj+m2+3]
    vpca2 = test_recons_reshaped[:, jj*m2+3:(jj+1)*m2+3]

    #u1svd = upca @ np.diag(spca) @ vpca1.T
    u2svd = upca @ np.diag(spca) @ vpca2.T
    
    #snap_true = u1svd[0:nx*ny, j].reshape((nx, ny))
    snap_test = u2svd[0:nx*ny,jj].reshape((nx,ny)).T
    ax.imshow(snap_test,cmap='RdBu_r', interpolation='bilinear')

    ax.axis('off')
    plt.tight_layout()

plt.savefig('comp4.pdf')
plt.show()    
    
    
    


