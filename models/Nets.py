#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


from .masked_layers import Linear, Conv2d


def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)


class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(ResNet.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, plan, initializer, outputs=None):
        super(ResNet, self).__init__()
        outputs = outputs or 10

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNet.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        # self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_resnet_') and
                5 > len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                (int(model_name.split('_')[2]) - 2) % 6 == 0 and
                int(model_name.split('_')[2]) > 2)

    @staticmethod
    def get_model_from_name(model_name, initializer=kaiming_normal, outputs=10):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.
        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.
        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.
        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """

        if not ResNet.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return ResNet(plan, initializer, outputs)


def cifar_resnet_20(num_classes=10):
    return ResNet.get_model_from_name('cifar_resnet_20', outputs=10)

