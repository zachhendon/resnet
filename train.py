import torch
import torch.nn as nn
from data.dataset import data_loader
from model import ResNet
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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

def train(model, loss_fn, optimizer, scheduler, epochs, model_name):
    best_val_loss = torch.inf
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"models/{model_name}_{timestamp}")

    print("Starting training...")

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


train_loader, val_loader = data_loader(
    data_dir='./data/datasets', batch_size=128,
    subset_ratio=0.25
)
num_training = sum(len(inputs) for inputs, _ in train_loader)
num_val = sum(len(inputs) for inputs, _ in val_loader)

model = ResNet().cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.25)
EPOCHS = 180

model_name = 'resnet-25'

train(model, loss_fn, optimizer, scheduler, EPOCHS, model_name)
