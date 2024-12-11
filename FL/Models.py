import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
    )


class UNet_client(nn.Module):
    def __init__(self):
        super(UNet_client, self).__init__()

        # DenseNet121
        densenet = models.densenet121(weights="DenseNet121_Weights.DEFAULT")

        # Encoder
        self.encoder0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.encoder1 = densenet.features.denseblock1
        self.encoder2 = densenet.features.transition1
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.encoder5 = conv_block(512, 1024)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )
        self.decoder1 = conv_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.decoder2 = conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.decoder3 = conv_block(256, 128)

        self.upconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.decoder4 = conv_block(320, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        e0 = F.relu(self.encoder0(x))
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        e5 = self.encoder5(self.pool(e4))

        # Decoder
        d1 = self.upconv1(e5)
        d1 = F.interpolate(d1, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat((d1, e4), dim=1)
        d1 = self.decoder1(d1)

        d2 = self.upconv2(d1)
        d2 = F.interpolate(d2, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.decoder2(d2)

        d3 = self.upconv3(d2)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.decoder3(d3)

        d4 = self.upconv4(d3)
        d4 = F.interpolate(d4, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.decoder4(d4)

        out = self.final_conv(d4)
        out = F.interpolate(out, size=(240, 240), mode="bilinear", align_corners=False)

        return torch.sigmoid(out)
    
class UNet_CT(UNet_client):
    pass

class UNet_MRI(UNet_client):
    pass

class UNet_US(UNet_client):
    pass


# TODO: add student model