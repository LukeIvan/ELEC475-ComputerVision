import torch.nn as nn

KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential (
            nn.Conv2d(3, 64, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),

            nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),

            nn.Conv2d(128, 256, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        )

        self.regressor = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256*4*4)
        x = self.regressor(x)

        return x
