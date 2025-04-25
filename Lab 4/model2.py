import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

class DMadNet(nn.Module):
    def __init__(self, n):
        super(DMadNet, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        
        # Save intermediate layers for skip connections
        self.firstconv = resnet18.conv1
        self.firstbn = resnet18.bn1
        self.firstrelu = resnet18.relu
        self.firstmaxpool = resnet18.maxpool
        self.encoder1 = resnet18.layer1
        self.encoder2 = resnet18.layer2
        self.encoder3 = resnet18.layer3
        self.encoder4 = resnet18.layer4

        # Decoder blocks with skip connections
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256 + 256, 128)  # +256 for skip connection
        self.decoder2 = DecoderBlock(128 + 128, 64)   # +128 for skip connection
        self.decoder1 = DecoderBlock(64 + 64, 32)     # +64 for skip connection
        
        # Final layers
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv2 = nn.Conv2d(32, n, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path with stored intermediate outputs
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        
        x2 = self.firstmaxpool(x1)
        e1 = self.encoder1(x2)    # Save for skip connection
        e2 = self.encoder2(e1)    # Save for skip connection
        e3 = self.encoder3(e2)    # Save for skip connection
        e4 = self.encoder4(e3)

        # Decoder path with skip connections
        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))

        # Final convolutions
        out = self.final_conv1(d1)
        out = self.final_conv2(out)
        
        # Final upsampling to match input size
        out = nn.functional.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 2)
        
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 2, 
            in_channels // 2, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        
        self.conv3 = nn.Conv2d(in_channels // 2, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x