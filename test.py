import torch
from model import ResNet
from data.dataset import data_loader
import torch.nn as nn
import argparse
from model import *

# Get model type and path from args
parser = argparse.ArgumentParser(description="test.py is the program to test a given ResNet model on the CIFAR-10 dataset")
models = {
    "resnet18": get_resnet18,
    "resnet34": get_resnet34
}
parser.add_argument('-m', '--model',
                    action='store',
                    type=str,
                    choices=models.keys(),
                    default='resnet18',
                    help="the model to train - resnet18 (default) or resnet34",
                    metavar="")
parser.add_argument('-f', '--file',
                    action='store',
                    type=str,
                    help="the name of the model to test",
                    metavar="",
                    required=True)
args = parser.parse_args()

# Load model
model_name = args.model
model_fn = models[model_name]
model = model_fn()

# Load saved model weights
model_path = args.file
model.load_state_dict(torch.load(f"models/{model_path}/best_model"))

# Load test data
test_loader = data_loader(
    data_dir='./data/datasets', batch_size=128,
    test=True
)
num_test = sum(len(inputs) for inputs, _ in test_loader)

# Evaluate data
num_correct = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        class_outputs = torch.argmax(outputs, dim=1)
        num_correct += (class_outputs == labels).sum()

test_acc = num_correct / num_test
print(f"Test accuracy: {test_acc:.4f}")
