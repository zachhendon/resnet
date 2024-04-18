import torch
from data.dataset import data_loader
from model import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse

def train_epoch(model, loss_fn, optimizer, scheduler):
    running_loss = 0
    num_correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        running_loss += loss.item() * len(inputs)

        class_outputs = torch.argmax(outputs, dim=1)
        num_correct += (class_outputs == labels).sum()

        optimizer.step()
    scheduler.step()

    avg_loss = running_loss / num_training
    avg_acc = num_correct / num_training
    return avg_loss, avg_acc

def val_epoch(model, loss_fn):
    running_loss = 0
    num_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * len(inputs)

            class_outputs = torch.argmax(outputs, dim=1)
            num_correct += (class_outputs == labels).sum()

        avg_loss = running_loss / num_val
        avg_acc = num_correct / num_val
        return avg_loss, avg_acc

def train(model, loss_fn, optimizer, scheduler, epochs, model_name, writer):
    best_val_loss = torch.inf

    for epoch in range(epochs):
        model.train(True)
        avg_train_loss, avg_train_acc = train_epoch(model, loss_fn, optimizer, scheduler)
        model.eval()
        avg_val_loss, avg_val_acc = val_epoch(model, loss_fn)
        print(f"[Epoch {epoch+1}/{epochs}]: train-loss = {avg_train_loss:.4f} | train-acc = {avg_train_acc:.4f} "
              f"| val-loss = {avg_val_loss:.4f} | val-acc = {avg_val_acc:.4f}")


        writer.add_scalars('Training vs Validation Loss',
                           {'Training': avg_train_loss, 'Validation': avg_val_loss},
                           epoch + 1)
        writer.add_scalars('Training vs Validation Accuracy',
                           {'Training': avg_train_acc, 'Validation': avg_val_acc},
                           epoch + 1)
        writer.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"models/{model_name}_{timestamp}/checkpoint_{epoch}"
            torch.save(model.state_dict(), model_path)

    best_model_path = f"models/{model_name}_{timestamp}/best_model"
    torch.save(model.state_dict(), best_model_path)
    print(f"Saved best model to {best_model_path}")

# Get model and size of dataset from args
parser = argparse.ArgumentParser(description="train.py is the program to train the CIFAR-10 dataset on a ResNet model")
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
def size_range(value):
    fvalue = float(value)
    if fvalue < 0.01 or fvalue > 1:
        raise argparse.ArgumentTypeError("%r is not in the range [0.01, 1]" % value)
    return fvalue
parser.add_argument('-s', '--size',
                    action='store',
                    default=1,
                    type=size_range,
                    help="the percent of the dataset to use for training - range=[0.01, 1] (default=1)",
                    metavar="")
args = parser.parse_args()
model_name = args.model
subset_ratio = args.size
print(f"Selected model: {model_name}")

# Load data
train_loader, val_loader = data_loader(
    data_dir='./data/datasets', batch_size=128,
    subset_ratio=subset_ratio
)
num_training = sum(len(inputs) for inputs, _ in train_loader)
num_val = sum(len(inputs) for inputs, _ in val_loader)

# Load model
model_fn = models[model_name]
model = model_fn()

# Initialize loss, optimizer, and schedule
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.25)
EPOCHS = 180

# Begin training
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f"models/{model_name}_{timestamp}")

print(f"Starting training for {model_name}_{timestamp}")
train(model, loss_fn, optimizer, scheduler, EPOCHS, model_name, writer)
