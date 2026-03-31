'''
Built code based on the paper below.

T. Zeng, C. Diao and D. Lu, "U-Net-Based Multispectral Image Generation From an RGB Image," in IEEE Access, vol. 9, pp. 43387-43396, 2021, doi: 10.1109/ACCESS.2021.3066472.
keywords: {Feature extraction;Spatial resolution;Neural networks;Dictionaries;Image synthesis;Image reconstruction;Training;Convolutional neural networks;multispectral image;U-Net},

'''

import torch.nn as nn
import torch

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
            nn.PReLU(out_channels,0.2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
            nn.PReLU(out_channels,0.2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
            nn.PReLU(out_channels,0.2)
        )
        
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
            nn.PReLU(out_channels,0.2)
        )
        
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        
    def forward(self, x):
        out_block1 = self.block1(x)
        out_block2 = self.block2(x)
        out_block3 = self.block3(x)
        out_block4 = self.block4(x)
        
        concat_block = torch.cat([out_block1, out_block2, out_block3, out_block4], dim=1)
        final_out = self.final_conv(concat_block)
        return final_out

class BaseUNet(nn.Module):
    def __init__(self):
        super(BaseUNet, self).__init__()
        
        # Encoder
        self.encoder_conv1 = InceptionBlock(3, 64)
        self.encoder_conv1_1 = InceptionBlock(64, 64)
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = InceptionBlock(64, 128)
        self.encoder_conv2_1 = InceptionBlock(128, 128)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = InceptionBlock(128, 256)
        self.encoder_conv3_1 = InceptionBlock(256, 256)
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = InceptionBlock(256, 512)
        self.encoder_conv4_1 = InceptionBlock(512, 512)
        self.encoder_pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.encoder_conv5 = InceptionBlock(512, 1024)
        self.encoder_conv5_1 = InceptionBlock(1024, 1024)
        
        # Decoder
        self.decoder_upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv1 = InceptionBlock(1024, 512)
        self.decoder_conv1_1 = InceptionBlock(512, 512)
        
        self.decoder_upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv2 = InceptionBlock(512, 256)
        self.decoder_conv2_1 = InceptionBlock(256, 256)
        
        self.decoder_upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv3 = InceptionBlock(256, 128)
        self.decoder_conv3_1 = InceptionBlock(128, 128)
        
        self.decoder_upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv4 = InceptionBlock(128, 64)
        self.decoder_conv4_1 = InceptionBlock(64, 64)
        
        
        # Output
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1),
            nn.PReLU(8,0.2)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)
        x1 = self.encoder_conv1_1(x1)
        x1_pool = self.encoder_pool1(x1)
        x2 = self.encoder_conv2(x1_pool)
        x2 = self.encoder_conv2_1(x2)
        x2_pool = self.encoder_pool2(x2)
        x3 = self.encoder_conv3(x2_pool)
        x3 = self.encoder_conv3_1(x3)
        x3_pool = self.encoder_pool3(x3)
        x4 = self.encoder_conv4(x3_pool)
        x4 = self.encoder_conv4_1(x4)
        x4_pool = self.encoder_pool4(x4)
        x5 = self.encoder_conv5(x4_pool)
        x5 = self.encoder_conv5_1(x5)
        
        # Decoder
        x = self.decoder_upsample1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder_conv1(x)
        x = self.decoder_conv1_1(x)
        
        x = self.decoder_upsample2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder_conv2(x)
        x = self.decoder_conv2_1(x)
        
        x = self.decoder_upsample3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder_conv3(x)
        x = self.decoder_conv3_1(x)
        
        
        x = self.decoder_upsample4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder_conv4(x)
        x = self.decoder_conv4_1(x)
        
        # Output
        x = self.output_conv(x)
        return x
