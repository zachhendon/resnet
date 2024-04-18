import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 2 if downsample else 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer2 = self._make_layer(64, num_blocks[0], downsample=False)
        self.layer3 = self._make_layer(128, num_blocks[1], downsample=True)
        self.layer4 = self._make_layer(256, num_blocks[2], downsample=True)
        self.layer5 = self._make_layer(512, num_blocks[3], downsample=True)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def _make_layer(self, out_channels, num_blocks, downsample):
        blocks = [ResidualBlock(self.in_channels, out_channels, downsample=downsample)]
        self.in_channels = out_channels

        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(self.in_channels, out_channels, downsample=False))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

def get_resnet18():
    model = ResNet([2, 2, 2, 2])
    model = nn.DataParallel(model)
    return model

def get_resnet34():
    model = ResNet([3, 4, 6, 3])
    model = nn.DataParallel(model)
    return model
