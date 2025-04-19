import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, group=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print('in block')
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, group=group)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, inchannel=270, activity_num=512):
        super(ResNet, self).__init__()
        # B*270*1000
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, activity_num)
        

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output = self.fc(output)

        return output


class ResNetLargeBert1(nn.Module):
    '''bert并行输出1024和55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetLargeBert1, self).__init__()
        # B*270*1000
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, activity_num)
        self.fc_bert = nn.Linear(512 * block.expansion, 1024)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output_bert = self.fc_bert(output)
        output = self.fc(output)

        return output, output_bert


class ResNetLargeBert2(nn.Module):
    '''bert串行输出: 512->1024->55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetLargeBert2, self).__init__()
        # B*270*1000
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_bert = nn.Linear(512 * block.expansion, 1024)
        self.fc = nn.Linear(1024, activity_num)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output_bert = self.fc_bert(output)
        output = self.fc(output_bert)

        return output, output_bert


class ResNetLargeBert3(nn.Module):
    '''bert pooling: 1024->55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetLargeBert3, self).__init__()
        # B*270*1000
        self.inplanes = 256
        self.conv1 = nn.Conv1d(inchannel, 256, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(1024, 1024, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024 * block.expansion, activity_num)
       

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output_bert = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output = self.fc(output_bert)

        return output, output_bert


class ResNetLargeBert4(nn.Module):
    '''bert pooling: 1024->55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetLargeBert4, self).__init__()
        # B*270*1000
        self.inplanes = 256
        self.conv1 = nn.Conv1d(inchannel, 256, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, group=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_w2v = nn.AdaptiveAvgPool1d(1024)  # B*1024
        self.fc = nn.Linear(1024, activity_num)
       

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output = output.view(output.size(0), -1)
        output_bert = self.avg_pool_w2v(output)

        # print(output.shape)
        # B*55
        output = self.fc(output_bert)

        return output, output_bert


class ResNetBaseBert1(nn.Module):
    '''bert并行输出1024和55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetBaseBert1, self).__init__()
        # B*270*1000
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, activity_num)
        self.fc_bert = nn.Linear(512 * block.expansion, 768)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output_bert = self.fc_bert(output)
        output = self.fc(output)

        return output, output_bert
      
class ResNetBaseBert2(nn.Module):
    '''bert串行输出: 512->768->55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetBaseBert2, self).__init__()
        # B*270*1000
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_bert = nn.Linear(512 * block.expansion, 768)
        self.fc = nn.Linear(768, activity_num)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output_bert = self.fc_bert(output)
        output = self.fc(output_bert)

        return output, output_bert      

class ResNetBaseBert3(nn.Module):
    '''bert pooling: 768->55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetBaseBert3, self).__init__()
        # B*270*1000
        self.inplanes = 192
        self.conv1 = nn.Conv1d(inchannel, 192, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(192)
        self.conv2 = nn.Conv1d(192, 192, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(192, 192, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(192)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 192, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 192, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 384, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 768, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(768, 768, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(768 * block.expansion, activity_num)
       

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output_bert = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output = self.fc(output_bert)

        return output, output_bert

class ResNetPollingWord2Vec(nn.Module):
    '''bert pooling: 768->55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetPollingWord2Vec, self).__init__()
        # B*270*1000
        self.inplanes = 37
        self.conv1 = nn.Conv1d(inchannel, 37, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(37)
        self.conv2 = nn.Conv1d(37, 37, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(192)
        self.conv3 = nn.Conv1d(37, 37, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(37)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 37, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 75, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 150, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 300, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(300, 300, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(300 * block.expansion, activity_num)
       

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('-------------------------------------')
        # B*270*1000
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        # B*128*500
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # print(x.shape)
        # B*128*2500
        x = self.maxpool(x)
        # print(x.shape)
        # B*128*1250
        c1 = self.layer1(x)
        # print(c1.shape)
        # B*128*1250
        c2 = self.layer2(c1)
        # print(c2.shape)
        # B*256*625
        c3 = self.layer3(c2)
        # print(c3.shape)
        # B*512*312
        c4 = self.layer4(c3)
        # print(c4.shape)
        # B*512*156
        # c4 = self.conv4(c4)
        # print(c4.shape)
        # B*512*1
        output = self.avg_pool(c4)
        # print(output.shape)
        # B*512
        output_bert = output.view(output.size(0), -1)
        # print(output.shape)
        # B*55
        output = self.fc(output_bert)

        return output, output_bert
     
       
def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet18_mutual():
    """ return a ResNet 18 object
    """
    return ResNetLargeBert3(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet34_mutual():
    """ return a ResNet 34 object
    """
    return ResNetLargeBert3(BasicBlock, [3, 4, 6, 3])


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet50_mutual():
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNetLargeBert4(Bottleneck, [3, 4, 6, 3])


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet101_mutual():
    """ return a ResNet 101 object
    """
    return ResNetLargeBert4(Bottleneck, [3, 4, 23, 3])


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(Bottleneck, [3, 8, 36, 3])


net = resnet18()
# # # model = lstm('lstm_es', data_type='rfid', seq_len=288, axis=23)
# x = torch.from_numpy(np.random.randint(0, 255, (10, 270, 2048))).float()
# # # # x2 = torch.from_numpy(np.random.randint(0, 255, (10, 23, 288))).float()
# y = net(x)
# print(y.shape)