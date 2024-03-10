import torch
import torch.optim as optim
from tqdm import tqdm
from models import Generator, Discriminator
from dataset import get_dataloader

# Hyperparameters
batch_size = 32
image_size = 64
nc = 3  # Number of channels in the training images. For color images this is 3
nz = 100  # Size of z latent vector (i.e., size of generator input)
ngf = 64  # Size of feature maps in the generator
ndf = 64  # Size of feature maps in the discriminator   
num_epochs = 5
lr = 0.0002
beta1 = 0.5

# Create the dataloader
dataloader = get_dataloader("hand_poses_dataset", batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device for training

if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
else:
    print('Device selected: CPU')

# Create the generator
netG = Generator(nz, ngf, nc).to(device)

# Create the Discriminator
netD = Discriminator(nc, ndf).to(device)  # Removed ndf

# Initialize BCELoss function
criterion = torch.nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # Create a progress bar
    progress_bar = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    # For each batch in the dataloader
    for i, data in progress_bar:
        # Move the input data to the device
        inputs = data[0].to(device)

        # Keep the labels in CPU
        labels = data[1]

        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = inputs
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(0)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Update the progress bar
        progress_bar.set_postfix({'Loss_D': f'{errD.item():.4f}', 'Loss_G': f'{errG.item():.4f}', 'D(x)': f'{D_x:.4f}', 'D(G(z))': f'{D_G_z1:.4f} / {D_G_z2:.4f}'})

        iters += 1

    print("Training Loop Finished")

 # Save the Discriminator model
torch.save(netD.state_dict(), 'discriminator.pth')