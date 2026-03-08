import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super().__init__()

        # convolutional blocks: two consecutive 3×3 convolutional layers followed by ReLU activation functions.
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3), #Conv 3×3
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3),
                nn.ReLU(inplace=True)
            )

        self.pool = nn.MaxPool2d(2) # reducing the spatial resolution through max-pooling operations

        # encoder feature goes from 64 to 256
        self.enc1 = CBR(in_channels, 64) #1
        self.enc2 = CBR(64, 128) #2
        self.enc3 = CBR(128, 256) # 3

        # bottleneck
        self.middle = CBR(256, 512)

        # decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2) #2x2
        self.dec3 = CBR(512, 256) #3

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128) #2

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64) #1

        self.out = nn.Conv2d(64, out_channels, 1) #1x1 

    def crop(self, enc_feat, x):
        _, _, H, W = x.shape
        return enc_feat[:, :, :H, :W]

    def forward(self, x):

        input_size = (x.shape[2], x.shape[3])

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.middle(self.pool(e3))

        # ---- 3
        u3 = self.up3(m)
        e3 = self.crop(e3, u3) # skip connection
        d3 = self.dec3(torch.cat([u3, e3], dim=1)) # concatenation encoder + decoder

        # ---- 2
        u2 = self.up2(d3)
        e2 = self.crop(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        # ---- 1
        u1 = self.up1(d2)
        e1 = self.crop(e1, u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))


        out = self.out(d1) # 1x1 = 64 channels → NUM_CLASSES

        # bilinear interpolation is applied to resize the output feature maps
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)