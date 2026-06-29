"""
ResNet45 with Triplet Deep Attention (TDA) module.
Modified from ABINet's ResNet45: TDA is inserted after the 3x3 conv + BN
in each BasicBlock, as described in Section 3.1.1 and Fig. 2 of the paper.
"""
import math
import torch.nn as nn
from .tda import TDAModule


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockTDA(nn.Module):
    """BasicBlock with TDA module inserted after 3x3 conv + BN + ReLU.
    
    Structure per paper Fig. 2:
      1x1 Conv -> BN & ReLU -> 3x3 Conv -> BN & ReLU -> TDA -> (+ residual) -> out
    
    Note: The paper shows TDA after BN&ReLU of the 3x3 conv. The ABINet original
    block applies ReLU after the residual add. We follow the paper's diagram where
    TDA sits between the 3x3 conv output and the residual connection.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # TDA module after 3x3 conv + BN (before residual add)
        self.tda = TDAModule(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply TDA after 3x3 conv + BN
        out = self.tda(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet45TDA(nn.Module):
    """ResNet45 with TDA modules.
    
    Architecture: 1 conv layer (kernel 3) + 5 stages with [3, 4, 6, 6, 3] blocks.
    Channels: [32, 64, 128, 256, 512] (from ABINet source).
    Strides: [2, 1, 2, 1, 1] (from ABINet source).
    """
    def __init__(self):
        super().__init__()
        self.inplanes = 32
        block = BasicBlockTDA
        layers = [3, 4, 6, 6, 3]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
