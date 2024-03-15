import torch
import torch.optim as optim
from tqdm import tqdm
from cnn import HandGestureCNN
from dataset import HandPoseDataset, get_dataloader
from torch.utils.data import random_split

# Hyperparameters
batch_size = 32
nc = 3  # Number of channels in the training images. For color images this is 3
ndf = 64  # Size of feature maps in the discriminator
num_epochs = 5
# num_epochs = 10
lr = 0.0002
beta1 = 0.5
# beta1 = 0.7
num_classes = 7

# # Create the dataloader
dataloader = get_dataloader("hand_poses_dataset_brutto", batch_size)

###WITH THE RANDOM SPLIT AND EARLY STOPPING###
####################################################
# # Create the dataset
# print("Creating dataset...")
# dataset = HandPoseDataset("hand_poses_dataset_cnn")

# # Ensure the dataset has enough elements
# assert len(dataset) > batch_size, "The dataset is too small"

# # Split the dataset into training and validation sets
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# assert train_size > 0 and val_size > 0, "Train or validation set is empty"

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Create the dataloaders
# print("Creating dataloaders...")
# train_dataloader = get_dataloader(train_dataset, batch_size=batch_size)
# val_dataloader = get_dataloader(val_dataset, batch_size=batch_size)
####################################################

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # Define the device for training

if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
else:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

# Initialize the CNN model
netCNN = HandGestureCNN(nc, ndf, num_classes).to(device)

# Initialize CrossEntropyLoss function
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.NLLLoss()

# Setup Adam optimizer for the CNN
optimizerCNN = optim.Adam(netCNN.parameters(), lr=lr, betas=(beta1, 0.999))

# # Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerCNN, 'min')
# # Early stopping parameters
# patience = 20
# best_loss = None
# epochs_no_improve = 0

# Training Loop

# Lists to keep track of progress
losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # Create a progress bar
    progress_bar = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    # For each batch in the dataloader
    for i, data in progress_bar:
        # Move the input data to the device
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizerCNN.zero_grad()

        # Forward pass
        outputs = netCNN(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizerCNN.step()

        # Print statistics
        losses.append(loss.item())
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, num_epochs, i, len(dataloader), loss.item()))

        # Update the progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        iters += 1

    print("Epoch Finished")
    print(f"Loss: {loss.item():.4f}")

    ###WITH THE RANDOM SPLIT AND EARLY STOPPING###
    ####################################################
    # # Validation loop
    # netCNN.eval()
    # with torch.no_grad():
    #     val_loss = 0
    #     for i, data in enumerate(val_dataloader, 0):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         outputs = netCNN(inputs)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()
    #     val_loss /= len(val_dataloader)

    # # Update the learning rate
    # scheduler.step(val_loss)

    # # Check for improvement
    # if best_loss is None or val_loss < best_loss:
    #     best_loss = val_loss
    #     epochs_no_improve = 0
    #     torch.save(netCNN.state_dict(), 'cnn_model_best.pth')
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve == patience:
    #         print("Early stopping")
    #         break
    ####################################################
# Save the CNN model
print("Saving the CNN model")
torch.save(netCNN.state_dict(), 'cnn_model.pth')
# torch.save(netCNN.state_dict(), 'cnn_model_randomSplit.pth')
