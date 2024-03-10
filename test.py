import torch
from dataset import get_dataloader
from models import Discriminator  # Ensure this matches your definition in models.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the test dataset
test_dataset = datasets.ImageFolder(root='hand_test_dataset')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def evaluate_discriminator(discriminator, dataloader, device):
    print("Starting test process...")
    discriminator.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Testing")
    with torch.no_grad():  # No need to track gradients
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = discriminator(images)
            predicted = torch.where(torch.abs(outputs) < 0.5, labels, torch.zeros_like(labels)).float()

            # predicted = (labels if outputs > 0.5 else 0).float()
            print(f"Predicted: {predicted}")
            print(f"Labels: {labels}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy of the Discriminator on the test images: {accuracy:.2f}%')
    print("Finished test process.")

# Load the test dataset
test_dataloader = get_dataloader("hand_test_dataset", batch_size=32)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
nc = 3  # Number of channels in the training images. For color images this is 3
ndf = 64  # Size of feature maps in the discriminator

# Load the trained Discriminator
netD = Discriminator(nc, ndf).to(device)  # Ensure this matches your model definition
netD.load_state_dict(torch.load('discriminator.pth'))

# Evaluate the Discriminator
evaluate_discriminator(netD, test_dataloader, device)
