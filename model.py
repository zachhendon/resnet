import torch.nn as nn

def get_layer(in_channels, out_channels, num_blocks):
    downsample = in_channels != out_channels
    blocks = [ResidualBlock(in_channels, out_channels, downsample=downsample)]

    for _ in range(num_blocks - 1):
        blocks.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*blocks)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
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

        self.downsample = downsample
        if downsample:
            self.downsampleLayer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsampleLayer(x)
        out += residual
        out = nn.ReLU()(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.layer2 = get_layer(16, 16, 9)
        self.layer3 = get_layer(16, 32, 9)
        self.layer4 = get_layer(32, 64, 9)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x
