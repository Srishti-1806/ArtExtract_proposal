'''
Adopted SegFormer architecture's encoder module to capture the coasrse and fine details of the image.

@misc{xie2021segformersimpleefficientdesign,
      title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers}, 
      author={Enze Xie and Wenhai Wang and Zhiding Yu and Anima Anandkumar and Jose M. Alvarez and Ping Luo},
      year={2021},
      eprint={2105.15203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2105.15203}, 
'''

import torch
import torch.nn as nn
from transBlocks.seg import mit_b1

#Applied SegTransformer encoder section
class SegUNet(nn.Module):
    def __init__(self):
        super(SegUNet, self).__init__()

        # Encoder - replace with MixVisionTransformer
        self.encoder = mit_b1()
        self.encoder_conv5 = Block1(512, 1024)
        self.encoder_conv5_1 = Block1(1024, 1024)

        # Decoder
        self.decoder_upsample1 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1)
        self.decoder_conv1 = Block2(1024, 512)
        self.decoder_conv1_1 = Block2(512, 512)
        
        self.decoder_upsample2 = nn.ConvTranspose2d(512, 320, kernel_size=2, stride=2)
        self.decoder_conv2 = Block2(640, 320) 
        self.decoder_conv2_1 = Block2(320, 320)
        
        self.decoder_upsample3 = nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2)  
        self.decoder_conv3 = Block2(256, 128)  
        self.decoder_conv3_1 = Block2(128, 128)
        
        self.decoder_upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.decoder_conv4 = Block2(128, 64)  
        self.decoder_conv4_1 = Block2(64, 64)
        
        # Output
        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(64,8,4,4,padding=0),
            nn.PReLU(8, 0.2)
        )

    def forward(self, x):
        encoder_outs = self.encoder.forward_features(x)
        # Assuming MixVisionTransformer returns 4 stages of outputs
        x1, x2, x3, x4 = encoder_outs[-4:] 
        
        # Decoder
        x5 = self.encoder_conv5(x4)
        x5 = self.encoder_conv5_1(x5)
        
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