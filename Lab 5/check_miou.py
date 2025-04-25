import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from model import DMadNet, DMadNetConfig

# Function to calculate mIoU
def calculate_miou(pred, target, num_classes=21):
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()

    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# Load the model architecture (replace with your model class)
config = DMadNetConfig(
        in_channels=3, num_classes=21, initial_filters=32, bilinear=True
)
model = DMadNet(config)

# Load the weights from the .pth file
weights_path = 'training_outputs/base_modely.pth'
model.load_state_dict(torch.load(weights_path)['model_state_dict'])
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Prepare the dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
])

dataset = VOCSegmentation(
    "./data",
    image_set="val",
    transform=transform,
    target_transform=mask_transform,
    download=False,
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Calculate mIoU for the entire dataset and visualize some predictions
total_miou = 0
num_samples = 0

with torch.no_grad():
    for idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        miou = calculate_miou(outputs, masks)
        average_miou = total_miou / num_samples
        
        print(f"Average mIoU: {average_miou:.4f}")

        # Visualize some predictions
        if idx < 5:  # Visualize first 5 samples
            pred_mask = torch.argmax(outputs[1], dim=1).squeeze(0).cpu().numpy()
            true_mask = masks.squeeze(0).cpu().numpy()
            
            fig, axarr = plt.subplots(1, 3)
            axarr[0].imshow(images.squeeze(0).permute(1, 2, 0).cpu().numpy())
            axarr[0].set_title('Input Image')
            axarr[1].imshow(true_mask)
            axarr[1].set_title('True Mask')
            axarr[2].imshow(pred_mask)
            axarr[2].set_title('Predicted Mask')
            plt.show()

        miou = calculate_miou(outputs, masks)
        
        total_miou += miou
        num_samples += 1

average_miou = total_miou / num_samples
print(f"Average mIoU: {average_miou:.4f}")
