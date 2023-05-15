import torchvision
import torch
import torch.nn as nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from livelossplot import PlotLosses
import copy
import time

import time
import os

def evaluate_model(model, testloader, model_name):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate unscripted model on test dataset
    unscripted_results = []
    with torch.no_grad():
        start = time.time()
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            unscripted_results.append(predicted.tolist())
        end = time.time()
        avg_time_unscripted = (end - start) / 100
        print("Average execution time for unscripted model: {:.6f} s".format(avg_time_unscripted))

    # Export model to TorchScript format and evaluate on test dataset
    scripted_model = torch.jit.script(model)
    scripted_model = scripted_model.to(device)
    scripted_results = []
    with torch.no_grad():
        start = time.time()
        for images, labels in testloader:
            images = images.to(device)
            outputs = scripted_model(images)
            predicted = torch.argmax(outputs, 1)
            scripted_results.append(predicted.tolist())
        end = time.time()
        avg_time_scripted = (end - start) / 100
        print("Average execution time for scripted model: {:.6f} s".format(avg_time_scripted))

    # Compare results between unscripted and scripted models
    num_correct = 0
    for unscripted_pred, scripted_pred in zip(unscripted_results, scripted_results):
        if unscripted_pred == scripted_pred:
            num_correct += 1

    accuracy = num_correct / len(testloader.dataset)

    # Save TorchScript model
    scripted_model_file = model_name + '_scripted.pt'
    scripted_model.save(scripted_model_file)
    print("Scripted model saved to {}".format(scripted_model_file))

    model_file_unscripted = model_name + '_unscripted.pt'
    torch.save(model, model_file_unscripted)
    model_size_unscripted = round(os.path.getsize(model_file_unscripted) / (1024 * 1024), 2)
    print("Unscripted model size: {} MB".format(model_size_unscripted))

    scripted_model_size = round(os.path.getsize(scripted_model_file) / (1024 * 1024), 2)
    print("Scripted model size: {} MB".format(scripted_model_size))

    # model_size_unscriptedd = get_model_size(model)
    # print("Model size for unscripted model: {:.2f} MB".format(model_size_unscriptedd))

    # model_size_scriptedd = get_model_size(scripted_model)
    # print("Model size for scripted model: {:.2f} MB".format(model_size_scriptedd))

    return accuracy
