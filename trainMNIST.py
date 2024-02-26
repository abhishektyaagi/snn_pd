import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from maskGenerator import get_mask_pseudo_diagonal_numpy

parser = argparse.ArgumentParser(description='Process a file.')
parser.add_argument('file_name', type=str, help='The name of the file to process')
args = parser.parse_args()

class MnistNetworkFC(nn.Module):
    def __init__(self):
        super(MnistNetworkFC, self).__init__()
        self.layer1 = nn.Linear(784, 300)
        self.layer2 = nn.Linear(300, 100)
        self.layer3 = nn.Linear(100, 10)
        
        # Initialize masks with ones (indicating non-zero weights)
        mask1_np = get_mask_pseudo_diagonal_numpy((300, 784), sparsity=0.97, file_name = str(args.file_name))
        mask2_np = get_mask_pseudo_diagonal_numpy((100, 300), sparsity=0.97, file_name = str(args.file_name))
        mask3_np = get_mask_pseudo_diagonal_numpy((10, 100), sparsity=0.97, file_name = str(args.file_name))
        
        self.mask1 = nn.Parameter(torch.tensor(mask1_np, dtype=torch.float32))
        self.mask2 = nn.Parameter(torch.tensor(mask2_np, dtype=torch.float32))
        self.mask3 = nn.Parameter(torch.tensor(mask3_np, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Function to calculate accuracy
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download MNIST dataset and define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download MNIST dataset
train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create an instance of the network and move it to the appropriate device
if torch.cuda.device_count() > 1:
  print(f"Let's use {torch.cuda.device_count()} GPUs!")
  model = nn.DataParallel(MnistNetworkFC())

model = model.to(device)

#Apply a mask to the weights
with torch.no_grad():
   model.module.layer1.weight.data *= model.module.mask1
   model.module.layer2.weight.data *= model.module.mask2
   model.module.layer3.weight.data *= model.module.mask3

# Set the initial mask values as desired
# For example, set certain elements to zero to sparsify the weights
# Example:
# model.mask1.data[desired_indices] = 0
# model.mask2.data[desired_indices] = 0
# model.mask3.data[desired_indices] = 0

# Define your optimizer
optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4)

# Define your loss function
criterion = nn.CrossEntropyLoss()

num_epochs = 15
log_interval = 1000
maxAcc = 0
# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    total_accuracy = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, 28*28).to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        # Update weights using masks
        with torch.no_grad():
            model.module.layer1.weight.grad *= model.module.mask1
            model.module.layer2.weight.grad *= model.module.mask2
            model.module.layer3.weight.grad *= model.module.mask3
            optimizer.step()

        #optimizer.step()

        ##TODO: Find a better way to do this
        #with torch.no_grad():
        #    model.module.layer1.weight.data *= model.module.mask1
        #    model.module.layer2.weight.data *= model.module.mask2
        #    model.module.layer3.weight.data *= model.module.mask3
  
        # Calculate accuracy
        acc = accuracy(outputs, target)
        total_accuracy += acc

        # Print statistics
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print("L1 Sparsity: ",np.count_nonzero(model.module.layer1.weight.data.cpu().numpy())/model.module.layer1.weight.data.numel())
    print("L2 Sparsity: ",np.count_nonzero(model.module.layer2.weight.data.cpu().numpy())/model.module.layer2.weight.data.numel())
    print("L3 Sparsity: ",np.count_nonzero(model.module.layer3.weight.data.cpu().numpy())/model.module.layer3.weight.data.numel())
    # Print epoch-level statistics
    print('\nEpoch {} completed. Average Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        epoch, total_loss / len(train_loader), total_accuracy / len(train_loader) * 100))
    if total_accuracy > maxAcc:
        maxAcc = total_accuracy / len(train_loader) * 100

with open('/p/dataset/abhishek/max_accuracy_'+str(args.file_name)+'.txt', 'a') as f:
    # Write the max accuracy to the file
    f.write(str(maxAcc))
    f.write("\n")
