import time
import os
import torch
import onnxruntime
import locale
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

print(locale.getpreferredencoding())

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import sys

def get_model_size(model):
    # calculate size of model
    total_size = 0
    for param_name, param in model.named_parameters():
        total_size += sys.getsizeof(param.data.storage())  
    return total_size / (1024 * 1024) # convert to MB

def evaluate_onnx_model(model, testloader, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert model to ONNX format
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_input = dummy_input.to(device)
    input_names = ["input"]
    output_names = ["output"]
    onnx_file = model_name + ".onnx"
    torch.onnx.export(model, dummy_input, onnx_file, verbose=False, input_names=input_names, output_names=output_names, export_params=True)

    # Load ONNX model in ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_file)

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
    model_size = round(os.path.getsize(onnx_file) / (1024 * 1024), 2)
    print("scripted model size: {} MB".format(model_size))

    # Report average execution time for ONNX model
    print("Average execution time for ONNX model: {:.6f} s".format(avg_time))

    # Measure execution time for PyTorch model
    num_correct = 0
    total_time_unscripted = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            start = time.time()
            _ = model(images)
            end = time.time()
            total_time_unscripted += (end - start)

            predicted = torch.argmax(_, 1)
            num_correct += (predicted == labels).sum().item()

    # Report average execution time for PyTorch model
    avg_time_unscripted = total_time_unscripted / len(testloader.dataset)
    print("Average execution time for unscripted model: {:.6f} s".format(avg_time_unscripted))

    # Measure unscripted model size
    model_size_unscripted = get_model_size(model)
    print("Unscripted model size: {:.2f} MB".format(model_size_unscripted))

    return accuracy
