import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x1 = x = self.classifier[:3](x)
        x2 = self.classifier[3:6](x)
        pending = torch.zeros(x.shape[0], 91 * 91 - 8192, device='cuda')
        return torch.cat((x1, x2, pending), dim=1).reshape((x1.shape[0], 1, 91, 91))


def alexnet(pretrained=False, device='cuda', **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
    del model.classifier[6]
    return model.to(device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=13, stride=2, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 96, kernel_size=11, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 21)
        )

    def forward(self, x):
        X = self.conv(x)
        x = self.classifier(x.flatten(start_dim=1))
        return x


def get_model(pretrained_backbone=True, device='cuda'):
    model = CNN().to(device)
    return alexnet(pretrained_backbone, device), model
