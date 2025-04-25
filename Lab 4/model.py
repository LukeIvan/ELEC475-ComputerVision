import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

class DMadNet(nn.Module):
    def __init__(self, n):
        super(DMadNet, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        
        kernel_size_upconv = 2
        padding_upconv = 1  # Adjusted padding
        kernel_size_conv = 3
        padding_conv = 1
        
        self.encoder = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,  # match resnet18 forward
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,  # bneck
        )
        
        self.decoder = nn.Sequential(
            # First block: 512 -> 384
            nn.ConvTranspose2d(512, 384, kernel_size=kernel_size_upconv, stride=2, padding=padding_upconv),
            #nn.Conv2d(384, 384, kernel_size=kernel_size_conv, padding=padding_conv),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            #nn.Dropout2d(0.1),
            
            # Second block: 384 -> 256
            nn.ConvTranspose2d(384, 256, kernel_size=kernel_size_upconv, stride=2, padding=padding_upconv),
            #nn.Conv2d(256, 256, kernel_size=kernel_size_conv, padding=padding_conv),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(0.1),
            
            # Third block: 256 -> 128
            nn.ConvTranspose2d(256, 128, kernel_size=kernel_size_upconv, stride=2, padding=padding_upconv),
            #nn.Conv2d(128, 128, kernel_size=kernel_size_conv, padding=padding_conv),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(0.1),
            
            # Fourth block: 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size_upconv, stride=2, padding=padding_upconv),
            #nn.Conv2d(64, 64, kernel_size=kernel_size_conv, padding=padding_conv),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(0.1),
            
            # Final classifier
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n, kernel_size=1),  # Remove LogSoftmax
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, input_tensor: Tensor):
        x = self.encoder(input_tensor)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x