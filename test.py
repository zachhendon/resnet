import torch
from model import ResNet
from data.dataset import data_loader

test_loader = data_loader(
    data_dir='./data/datasets', batch_size=128,
    test=True
)
num_test = sum(len(inputs) for inputs, _ in test_loader)

model = ResNet().cuda()
model.load_state_dict(torch.load('models/resnet-25_20240417_010445/checkpoint_91'))

num_correct = 0
model.eval().cuda()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        class_outputs = torch.argmax(outputs, dim=1)
        num_correct += (class_outputs == labels).sum()

test_acc = num_correct / num_test
print(f"Test accuracy: {test_acc:.4f}")
