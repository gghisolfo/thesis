import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
# in_channels: es. 3 if RGB
# out_channels: number of channels: paddle, ball, walls, ...

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch): #piccolo modulo composto da 2 convoluzioni 3x3
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), #BachNorm
                nn.ReLU(inplace=True) #RELu
            )

        #encoder
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)

        self.pool = nn.MaxPool2d(2)


        self.middle = CBR(256, 512)


        #decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1) #conv 1x1

    # def forward(self, x):
    #     e1 = self.enc1(x)
    #     e2 = self.enc2(self.pool(e1))
    #     e3 = self.enc3(self.pool(e2))

    #     m = self.middle(self.pool(e3))

    #     d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
    #     d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
    #     d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

    #     return self.out(d1)

    #forward pass
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.middle(self.pool(e3))

        u3 = self.up3(m)
        u3 = F.interpolate(u3, size=e3.shape[2:])  # Match height and width with e3
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)

