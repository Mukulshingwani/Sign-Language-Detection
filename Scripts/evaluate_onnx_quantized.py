import os 
import time
import onnxruntime

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

# Load the CIFAR100 dataset
trainset = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8) #QInt8

    print(f"quantized model saved to:{quantized_model_path}")

quantize_onnx_model('resnet34.onnx', 'resnet34_quant.onnx')

print('ONNX full precision model size (MB):', os.path.getsize("model.onnx")/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize("model_quant.onnx")/(1024*1024))

# Load ONNX model in ONNX Runtime
ort_session = onnxruntime.InferenceSession('resnet34_quant.onnx')

# Evaluate model on test dataset
num_correct = 0
start_time = time.time()
for images, labels in testloader:
    ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    predicted = torch.argmax(torch.tensor(ort_outs), 1)
    num_correct += (predicted == labels).sum().item()
end_time = time.time()
avg_time = (end_time - start_time) / len(testloader.dataset)

accuracy = num_correct / len(testloader.dataset)

# Measure model size
model_size = round(os.path.getsize('resnet34_quant.onnx') / (1024 * 1024), 2)
print("quant scripted model size: {} MB".format(model_size))

# Report average execution time for ONNX model
print("Average execution time for ONNX quant model: {:.6f} s".format(avg_time))