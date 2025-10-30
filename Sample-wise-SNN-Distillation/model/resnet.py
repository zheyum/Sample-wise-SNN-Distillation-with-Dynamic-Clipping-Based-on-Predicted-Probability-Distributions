# -*- coding: utf-8 -*-
from model.layer import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, expand=1, **kwargs_spikes):
        super(BasicBlock, self).__init__()
        self.expand = expand
        self.conv1 = nn.Conv2d(in_planes, planes * expand, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * expand)
        self.spike1 = LIFLayer(**kwargs_spikes)
        self.conv2 = nn.Conv2d(planes, planes * expand, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * expand)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion * expand, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes * expand)
            )
        self.spike2 = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.spike2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, **kwargs_spikes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            LIFLayer(**kwargs_spikes)
        )
        self.layer1 = self._make_layer(block, 64, num_block_layers[0], stride=1, **kwargs_spikes)
        self.layer2 = self._make_layer(block, 128, num_block_layers[1], stride=2, **kwargs_spikes)
        self.layer3 = self._make_layer(block, 256, num_block_layers[2], stride=2, **kwargs_spikes)
        self.layer4 = self._make_layer(block, 512, num_block_layers[3], stride=2, **kwargs_spikes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes),
        )

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs_spikes):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs_spikes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        out = self.conv0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)

        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        out = out.reshape(self.time_step, -1, out.shape[1])

        return out, feature 

class TimePooledConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, pool_type='avg'):
        super(TimePooledConvBlock, self).__init__()
        self.pool_type = pool_type

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        T = 2       
        C, H, W = x.shape[1],x.shape[2],x.shape[3]
        x = x.view(T, -1, C, H, W)  # ×Ô¶¯ÍÆ¶Ï batch_size

        if self.pool_type == 'avg':
            x = x.mean(dim=0)         # ¡ú [N, C, H, W]
        elif self.pool_type == 'max':
            x, _ = x.max(dim=0)       # ¡ú [N, C, H, W]
        else:
            raise ValueError("Unsupported pool_type: use 'avg' or 'max'")

        x = self.conv(x)
        x = self.bn(x)        
        x = self.relu(x) 
        return x
        
        
class ResNet19(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, **kwargs_spikes):
        super(ResNet19, self).__init__()
        self.in_planes = 128
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            LIFLayer(**kwargs_spikes)
        )
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], stride=1, **kwargs_spikes)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], stride=2, **kwargs_spikes)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], stride=2, **kwargs_spikes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * block.expansion, 256, bias=True),
            LIFLayer(**kwargs_spikes),
            nn.Linear(256, num_classes, bias=True),
        )
        self.TimePool_1 = TimePooledConvBlock(128,128)
        self.TimePool_2 = TimePooledConvBlock(256,256)



    def _make_layer(self, block, planes, num_blocks, stride, **kwargs_spikes):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs_spikes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        
        out = self.conv0(x)
        out = self.layer1(out)
        feature.append(out)
        
        out = self.layer2(out)
        feature.append(out)
        
        out = self.layer3(out)
        #feature.append(out)
        
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        out = out.reshape(self.time_step, -1, out.shape[1])
        return out, feature


def resnet18(num_classes=10, in_channel=3, neuron_dropout=0.0, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channel=in_channel, **kwargs)


def resnet19(num_classes=10, in_channel=3, neuron_dropout=0.0, **kwargs):
    return ResNet19(BasicBlock, [3, 3, 2], num_classes, in_channel=in_channel, **kwargs)
