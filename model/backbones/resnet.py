import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, return_before_act):
        super(ResBlock, self).__init__()
        self.return_before_act = return_before_act
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.ds    = nn.Sequential(*[
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(out_channels)
                            ])
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.ds    = None
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        pout = self.conv1(x) # pout: pre out before activation
        pout = self.bn1(pout)
        pout = self.relu(pout)

        pout = self.conv2(pout)
        pout = self.bn2(pout)

        if self.downsample:
            residual = self.ds(x)

        pout += residual
        out  = self.relu(pout)

        if not self.return_before_act:
            return out
        else:
            return pout, out


class ResNet_simple(BaseModel):
    def __init__(self, block, num_blocks, num_class = 10, init_weights = True, deg_flag = None, fa = True):
        super(ResNet_simple, self).__init__(deg_flag)
        self.block = block
        self.num_blocks = num_blocks
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU()

        self.res1 = self.make_layer(self.block, self.num_blocks[0], 16, 16)
        self.res2 = self.make_layer(self.block, self.num_blocks[1], 16, 32)
        self.res3 = self.make_layer(self.block, self.num_blocks[2], 32, 64)
 
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc      = nn.Linear(256, num_class)

        if init_weights:
            self._init_weight_layers(self)

        self.num_class = num_class
        self.fa = fa
    
    def make_layer(self, block, num, in_channels, out_channels): # num must >=2
        layers = [block(in_channels, out_channels, False)]
        for i in range(num-2):
            layers.append(block(out_channels, out_channels, False))
        layers.append(block(out_channels, out_channels, True))
        return nn.Sequential(*layers)

    def forward(self, *x):
        x = self.define_input(*x)
        pstem = self.conv1(x) # pstem: pre stem before activation
        pstem = self.bn1(pstem)
        stem  = self.relu(pstem)
        stem  = (pstem, stem)

        rb1 = self.res1(stem[1])
        rb2 = self.res2(rb1[1])
        rb3 = self.res3(rb2[1])

        feat = self.avgpool(rb3[1])
        feat = feat.view(feat.size(0), -1)
        out  = self.fc(feat)

        return stem, rb1, rb2, rb3, feat, out

def ResNet20(**args):
    return ResNet_simple(ResBlock, [3,3,3], **args)

def ResNet56(**args):
    return ResNet_simple(ResBlock, [9,9,9], **args)

def ResNet110(**args):
    return ResNet_simple(ResBlock, [18,18,18], **args)
