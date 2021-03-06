import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(32)
        self.block3 = ResidualBlock(32)
        self.block4 = ResidualBlock(32)
        self.block5 = ResidualBlock(32)
        self.block6 = ResidualBlock(32)
        self.block7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        #block8 = [UpsampleBLock(64, 10) for _ in range(upsample_block_num)]
        block8 = [UpsampleBLock(32, 10)]
        block8.append(nn.Conv2d(32, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)
        self.tempblock = nn.Conv2d(3,1,kernel_size = 6, padding = 0)
        self.landseablock = nn.Sequential(
                      nn.Conv2d(3,1,kernel_size = 6, padding = 0),
                      nn.Softmax()
                      )
        self.hightblock = nn.Conv2d(3,1, kernel_size = 6, padding = 0)
    def forward(self, x):
        block1  = self.block1(x)
        block2  = self.block2(block1)
        block3  = self.block3(block2)
        block4  = self.block4(block3)
        block5  = self.block5(block4)
        block6  = self.block6(block5)
        block7  = self.block7(block6)
        block8  = self.block8(block1 + block7)
        temp    = self.tempblock(block8)
        landsea = self.landseablock(block8)
        hight   = self.hightblock(block8)
        return  temp, landsea, hight

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        #return torch.sigmoid(self.net(x).view(batch_size, 1))
        return self.net(x).view(batch_size, 1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
        #self.upsample = nn.Upsample(scale_factor = up_scale, mode = 'bilinear', align_corners = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        #x = self.upsample(x)
        return x

