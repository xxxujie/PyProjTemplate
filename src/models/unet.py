import torch
import settings
from torch.utils import checkpoint
from torch import nn
from ..utils.loggers import MyLogger


logger = MyLogger(__name__, settings.LOG_PATH).get_logger()
logger.info("test")


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down_sample1 = DownSample(64)
        self.conv2 = DoubleConv(64, 128)
        self.down_sample2 = DownSample(128)
        self.conv3 = DoubleConv(128, 256)
        self.down_sample3 = DownSample(256)
        self.conv4 = DoubleConv(256, 512)
        self.down_sample4 = DownSample(512)
        self.conv5 = DoubleConv(512, 1024)
        self.up_sample1 = UpSample(1024)
        self.conv6 = DoubleConv(1024, 512)
        self.up_sample2 = UpSample(512)
        self.conv7 = DoubleConv(512, 256)
        self.up_sample3 = UpSample(256)
        self.conv8 = DoubleConv(256, 128)
        self.up_sample4 = UpSample(128)
        self.conv9 = DoubleConv(128, 64)
        self.out = OutConv(64, out_classes)

    def forward(self, x):
        run1 = self.conv1(x)
        run2 = self.conv2(self.down_sample1(run1))
        run3 = self.conv3(self.down_sample2(run2))
        run4 = self.conv4(self.down_sample3(run3))
        run5 = self.conv5(self.down_sample4(run4))
        run6 = self.conv6(self.up_sample1(run5))
        run7 = self.conv7(self.up_sample2(run6))
        run8 = self.conv8(self.up_sample3(run7))
        run9 = self.conv9(self.up_sample4(run8))
        final_run = self.out(run9)
        return final_run

    def use_checkpoint(self):
        """Save the memory, but increase the runnig time."""

        self.conv1 = checkpoint(self.conv1)
        self.down_sample1 = checkpoint(self.down_sample1)
        self.conv2 = checkpoint(self.conv2)
        self.down_sample2 = checkpoint(self.down_sample2)
        self.conv3 = checkpoint(self.conv3)
        self.down_sample3 = checkpoint(self.down_sample3)
        self.conv4 = checkpoint(self.conv4)
        self.down_sample4 = checkpoint(self.down_sample4)
        self.conv5 = checkpoint(self.conv5)
        self.up_sample1 = checkpoint(self.up_sample1)
        self.conv6 = checkpoint(self.conv6)
        self.up_sample2 = checkpoint(self.up_sample2)
        self.conv7 = checkpoint(self.conv7)
        self.up_sample3 = checkpoint(self.up_sample3)
        self.conv8 = checkpoint(self.conv8)
        self.up_sample4 = checkpoint(self.up_sample4)
        self.conv9 = checkpoint(self.conv9)


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channle=None):
        super().__init__()
        if not mid_channle:
            mid_channle = out_channel
        self.double_conv = nn.Sequential(
            # 因为后面接BatchNorm，所以bias没用
            nn.Conv2d(
                in_channel,
                mid_channle,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channle),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channle,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):

    def __init__(self, channel):
        super().__init__()
        # 最大池化没有特征提取能力，丢失特征太多
        # 这里改成用3x3，步长为2的卷积进行下采样
        self.down_sample = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="biliner", align_corners=True),
            nn.Conv2d(channel, channel // 2, kernel_size=1, padding=1, stride=1),
        )

    def forward(self, encoder_featuer_map):
        out = self.up_sample
        out = torch.cat([out, encoder_featuer_map], dim=1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
