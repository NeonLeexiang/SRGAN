"""
    date:       2021/4/7 2:59 下午
    written by: neonleexiang
"""
import torch
from torch.nn import Conv2d, PReLU, Sequential, BatchNorm2d, LeakyReLU, Module, PixelShuffle, AdaptiveAvgPool2d
import math


class ResidualBlock(Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = BatchNorm2d(channels)
        self.p_relu = PReLU()
        self.conv2 = Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.p_relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return residual + x


class UpSampleBlock(Module):
    def __init__(self, in_channels, up_scale):
        super(UpSampleBlock, self).__init__()
        self.conv = Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=(3, 3), padding=(1, 1))
        self.pixel_shuffle = PixelShuffle(up_scale)
        self.p_relu = PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.p_relu(x)
        return x


class Generator(Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = Sequential(
            Conv2d(3, 64, kernel_size=(9, 9), padding=(4, 4)),
            PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = Sequential(
            Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(64)
        )

        # up sampling
        block8 = [UpSampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(Conv2d(64, 3, kernel_size=(9, 9), padding=(4, 4)))
        self.block8 = Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        return (torch.tanh(block8) + 1) / 2


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1)),
            LeakyReLU(0.2),

            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64),
            LeakyReLU(0.2),

            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(128),
            LeakyReLU(0.2),

            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128),
            LeakyReLU(0.2),

            Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(256),
            LeakyReLU(0.2),

            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256),
            LeakyReLU(0.2),

            Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(512),
            LeakyReLU(0.2),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512),
            LeakyReLU(0.2),

            AdaptiveAvgPool2d(1),
            Conv2d(512, 1024, kernel_size=(1, 1)),
            LeakyReLU(0.2),
            Conv2d(1024, 1, kernel_size=(1, 1)),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
