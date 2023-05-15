import torch
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import torchvision  # for dataset
import torch.nn as nn   # for neural networks
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# Define transformations for data augmentation
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the CIFAR100 dataset
trainset = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# create a SummaryWriter for Tensorboard logging
log_dir = '/content/drive/MyDrive/Deep Learning Lab/Lab 6/Logs/resnet34'
writer = SummaryWriter(log_dir)

# define profiler options
profiler_opts = {
    'schedule': torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    'on_trace_ready': torch.profiler.tensorboard_trace_handler(log_dir),
    'record_shapes': True,
    'profile_memory': True,
    'with_stack': True
}

def train(model, data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model = ...
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
# train the model for 5 epochs
for epoch in range(5):
    # create profiler object
    profiler = torch.profiler.profile(**profiler_opts)
    # iterate over the training data
    for step, batch_data in enumerate(trainloader):
        if step >= 10:
            break
        train(model, batch_data)
        # step the profiler
        profiler.step()
    # flush the writer
    writer.flush()
    # close the profiler
    profiler.__exit__(None, None, None)

writer.close()
