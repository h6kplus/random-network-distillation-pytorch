import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock, Bottleneck

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResNet(torch_resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

    def modify(self, remove_layers=[], padding=''):
        # Set stride of layer3 and layer 4 to 1 (from 2)
        
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1)
        self.conv4=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1,stride=1)
        self.relu= nn.LeakyReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x        


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)