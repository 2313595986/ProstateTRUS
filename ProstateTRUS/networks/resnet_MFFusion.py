import math
import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F


class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = x * y.expand(x.size())
        return y



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        # self.SE_block = SEblock(channels * self.expansion)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)
        # out = self.SE_block(out)

        return out


class Encoder2_MFFusion(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(Encoder2_MFFusion, self).__init__()
        layers = [3, 4, 6, 3]
        self.in_channels = 64
        self.conv1_1 = nn.Conv3d(num_channels, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.bn1_1 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.layer1_1 = self._make_layer(Bottleneck, 64, layers[0], stride=1)
        self.layer1_2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer1_3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer1_4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)

        self.in_channels = 64
        self.conv2_1 = nn.Conv3d(num_channels, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.bn2_1 = nn.InstanceNorm3d(64)
        self.layer2_1 = self._make_layer(Bottleneck, 64, layers[0], stride=1)
        self.layer2_2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer2_3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer2_4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)

        self.FF1 = FeatureFusion(256)
        self.FF2 = FeatureFusion(512)
        self.FF3 = FeatureFusion(1024)
        self.FF4 = FeatureFusion(2048)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.dropout = nn.Dropout(0.4)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        c_init = x.size(1)
        x1 = x[:, : int(c_init / 2), ...]
        x2 = x[:, int(c_init / 2):, ...]

        layer1_1_in = self.max_pool(self.relu(self.bn1_1(self.conv1_1(x1))))
        layer2_1_in = self.max_pool(self.relu(self.bn2_1(self.conv2_1(x2))))

        layer1_1_out = self.layer1_1(layer1_1_in)
        layer2_1_out = self.layer2_1(layer2_1_in)
        fusion_out1 = self.FF1(layer1_1_out, layer2_1_out)

        layer1_2_in = layer1_1_out + fusion_out1
        layer2_2_in = layer2_1_out + fusion_out1
        layer1_2_out = self.layer1_2(layer1_2_in)
        layer2_2_out = self.layer2_2(layer2_2_in)
        fusion_out2 = self.FF2(layer1_2_out, layer2_2_out)

        layer1_3_in = layer1_2_out + fusion_out2
        layer2_3_in = layer2_2_out + fusion_out2
        layer1_3_out = self.layer1_3(layer1_3_in)
        layer2_3_out = self.layer2_3(layer2_3_in)
        fusion_out3 = self.FF3(layer1_3_out, layer2_3_out)

        layer1_4_in = layer1_3_out + fusion_out3
        layer2_4_in = layer2_3_out + fusion_out3
        layer1_4_out = self.layer1_4(layer1_4_in)
        layer2_4_out = self.layer2_4(layer2_4_in)
        fusion_out4 = self.FF4(layer1_4_out, layer2_4_out)

        # out1 = self.avg_pool(layer1_4_out).view(bs, -1)
        # out2 = self.avg_pool(layer2_4_out).view(bs, -1)
        out_fusion = self.avg_pool(fusion_out4).view(bs, -1)

        # out_cls1 = self.fc1(out1)
        # out_cls2 = self.fc2(out2)
        out_cls_fusion = self.fc_fusion(out_fusion)

        return out_cls_fusion


def init_conv_weights(layer, weights_std=0.01,  bias=0):
    nn.init.xavier_normal_(layer.weight)
    nn.init.constant_(layer.bias.data, val=bias)
    return layer


def conv1x1x1(in_channels, out_channels, **kwargs):
    layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)
    return layer


def conv3x3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer


class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = conv1x1x1(in_channels * 2, in_channels)
        self.conv2 = conv1x1x1(in_channels * 2, in_channels)
        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()

    def forward(self, f1, f2):
        avg_feature = torch.mean(torch.stack([f1, f2], dim=0), dim=0)
        attention_map1 = self.sigmod(self.bn1(self.conv1(torch.cat((f1, avg_feature), dim=1))))
        attention_map2 = self.sigmod(self.bn2(self.conv2(torch.cat((f2, avg_feature), dim=1))))

        attention_map_sm1 = torch.exp(attention_map1) / (torch.exp(attention_map1) + torch.exp(attention_map2))
        attention_map_sm2 = torch.exp(attention_map2) / (torch.exp(attention_map1) + torch.exp(attention_map2))

        attention_feature1 = f1 * attention_map_sm1
        attention_feature2 = f2 * attention_map_sm2
        fusion_attention_feature = attention_feature1 + attention_feature2
        return fusion_attention_feature



if __name__ == '__main__':
    import os
    import torch.nn.functional as F

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = Encoder2_MFFusion(num_channels=3, num_classes=2).cuda()
    # model = nn.DataParallel(model).cuda()
    model.train()
    input1 = torch.randn([1, 6, 144, 144, 200]).cuda()
    out = model(input1)
    out = F.softmax(out, 1)
    print(out)

