import torch
from torchvision import models, datasets
from model2 import DMadNet
from torchvision.transforms.functional import resize
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

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
            ious.append(float('nan'))  # Ignore this class if there's no ground truth
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def custom_collate(batch):
    images = []
    targets = []
    
    # Image transform
    img_transform = v2.Compose([
        v2.Resize((500, 500)),  # Consistent size
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    def process_target(target):
        target_array = np.array(target)
        
        target_resized = np.array(v2.Resize((500, 500), interpolation=v2.InterpolationMode.NEAREST)(target))
        
        return torch.as_tensor(target_resized, dtype=torch.long)
    
    for (img, target) in batch:
        # Transform image
        img_tensor = img_transform(img)
        # Transform target
        target_tensor = process_target(target)
        
        images.append(img_tensor)
        targets.append(target_tensor)
    
    return torch.stack(images), torch.stack(targets)

def get_voc_dataset(split="val"):
    dataset = datasets.VOCSegmentation(
        root="./data",
        year="2012",
        image_set=split,
        download=True,
        transforms=None 
    )
    return dataset

def evaluate_model(model, dataloader, device, num_classes=21):
    model.eval()
    miou_scores = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)['out']
            
            for i in range(len(images)):
                pred = outputs[i:i+1]
                target = targets[i:i+1]
                
                miou = calculate_miou(pred, target, num_classes)
                miou_scores.append(miou)
    
    return np.nanmean(miou_scores)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = models.segmentation.fcn_resnet50(pretrained=True).to(device)
    
    dataset = get_voc_dataset(split="val")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, 
                            num_workers=4, collate_fn=custom_collate)
    
    mean_iou = evaluate_model(model, dataloader, device)
    print(f"Mean IoU on PASCAL VOC 2012 validation set: {mean_iou:.4f}")

if __name__ == "__main__":
    main()