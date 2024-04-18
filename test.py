import torch
from model import ResNet
from data.dataset import data_loader
import torch.nn as nn

test_loader = data_loader(
    data_dir='./data/datasets', batch_size=128,
    test=True
)
num_test = sum(len(inputs) for inputs, _ in test_loader)

model = ResNet([3, 4, 6, 3]).cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('models/resnet34_20240417_173517/best_model'))

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
