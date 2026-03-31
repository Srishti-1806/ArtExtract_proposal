'''
Applied SERT(Spectral Enhanced Rectangle Transformer) at the end of each decoder blocks to denoise the output.

@misc{li2023spectralenhancedrectangletransformer,
      title={Spectral Enhanced Rectangle Transformer for Hyperspectral Image Denoising}, 
      author={Miaoyu Li and Ji Liu and Ying Fu and Yulun Zhang and Dejing Dou},
      year={2023},
      eprint={2304.00844},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.00844}, 
}

'''

import torch
import torch.nn as nn
from unets.transBlocks.sert import SERT

class Block1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block1, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
            nn.PReLU(out_channels,0.2)
        )
        
    def forward(self, x):
        out_block1 = self.block1(x)
        return out_block1
    
class Block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block2, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            # nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
            nn.PReLU(out_channels,0.2)
        )

    def forward(self, x):
        out_block1 = self.block1(x)
        return out_block1
        
#SERT for denoising in between decoder blocks
class SERTUnet(nn.Module):
    def __init__(self):
        super(SERTUnet, self).__init__()
        self.sert_block0 = SERT(inp_channel_index=0)
        self.sert_block1 = SERT(inp_channel_index=1)
        self.sert_block2 = SERT(inp_channel_index=2)
        self.sert_block3 = SERT(inp_channel_index=3)
        
        # Encoder
        self.encoder_conv1 = Block1(3, 64)
        self.encoder_conv1_1 = Block1(64, 64)
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = Block1(64, 128)
        self.encoder_conv2_1 = Block1(128, 128)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = Block1(128, 256)
        self.encoder_conv3_1 = Block1(256, 256)
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = Block1(256, 512)
        self.encoder_conv4_1 = Block1(512, 512)
        self.encoder_pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.encoder_conv5 = Block1(512, 1024)
        self.encoder_conv5_1 = Block1(1024, 1024)
        
        # Decoder
        self.decoder_upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv1 = Block2(1024, 512)
        self.decoder_conv1_1 = Block2(512, 512)
        
        self.decoder_upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv2 = Block2(512, 256)
        self.decoder_conv2_1 = Block2(256, 256)
        
        self.decoder_upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv3 = Block2(256, 128)
        self.decoder_conv3_1 = Block2(128, 128)
        
        self.decoder_upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv4 = Block2(128, 64)
        self.decoder_conv4_1 = Block2(64, 64)
        
        # Output
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=3, padding='same'),
            nn.PReLU(8,0.2)
        )
        # nn.Conv2d(64, 8, kernel_size=1)
        
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
        # Apply SERT block in different decoder locations
        x = self.decoder_upsample1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder_conv1(x)
        x = self.decoder_conv1_1(x)
        # x = self.sert_block0(x)
        
        x = self.decoder_upsample2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder_conv2(x)
        x = self.decoder_conv2_1(x)
        # x = self.sert_block1(x)
        
        x = self.decoder_upsample3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder_conv3(x)
        x = self.decoder_conv3_1(x)
        # x = self.sert_block2(x)
        
        x = self.decoder_upsample4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder_conv4(x)
        x = self.decoder_conv4_1(x)
        x = self.sert_block3(x)
        
        # Output
        x = self.output_conv(x)
        return x