'''

Inspired by the enhanced connectivity between encoder and decoder blocks in Unet2 and Unet3,
I implemented a sparsely connected U-Net. This architecture reduces the concatenation of 
encoder outputs with decoder inputs as the decoder block approaches its final layers.

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

class SparseUNet(nn.Module):
    def __init__(self):
        super(SparseUNet, self).__init__()
        
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
        self.encoder_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv5 = InceptionBlock(512, 1024)
        self.encoder_conv5_1 = InceptionBlock(1024, 1024)
        
        self.max8x = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
        self.max4x = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.max2x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3_1 = InceptionBlock(128,64)
        self.conv3_2 = InceptionBlock(256,64)
        self.conv3_3 = InceptionBlock(512,64)
        self.d_conv1 = InceptionBlock(320,320)
        self.d_conv2 = InceptionBlock(192,192)        
        
        self.up2x =  nn.ConvTranspose2d(1024, 64, kernel_size=2, stride=2)
        self.up8x =  nn.ConvTranspose2d(1024, 64, kernel_size=8, stride=8)
        self.up2x_1 =  nn.ConvTranspose2d(320, 64, kernel_size=2, stride=2)
        self.up2x_2 =  nn.ConvTranspose2d(512, 64, kernel_size=2, stride=2)
        self.up2x_3 =  nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2)
        self.up2x_4 =  nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)

        # Output
        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1),
            nn.PReLU(8, 0.2)
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
    
        #Decoder  
        #d1
        x1_64 = self.max8x(x1)
        x2_64 = self.max4x(x2)
        x2_64 = self.conv3_1(x2_64)
        
        x3_64 = self.max2x(x3)
        x3_64 = self.conv3_2(x3_64)
        
        x4_64 = self.conv3_3(x4)
        x5_64 = self.up2x(x5)
        d1 = torch.concat([x1_64,x2_64,x3_64,x4_64,x5_64],dim=1)
        d1 = self.d_conv1(d1)

        #d2
        d1_up = self.up2x_1(d1)
        x2_64 = self.max2x(x2)
        x2_64 = self.conv3_1(x2_64)
        x4_64 = self.up2x_2(x4)
        d2 = torch.concat([d1_up,x2_64,x4_64],dim=1)
        d2 = self.d_conv2(d2)
        
        #d3
        d2_up = self.up2x_3(d2)
        x3_64 = self.up2x_4(x3)
        x5_64 = self.up8x(x5)
        d3 = torch.concat([d2_up,x3_64,x5_64],dim=1)
        d3 = self.d_conv2(d3)
        
        #d4
        d3_up = self.up2x_3(d3)
        d4 = torch.cat([d3_up,x1],dim=1)
        
        out = self.output_conv(d4)
        
        return out