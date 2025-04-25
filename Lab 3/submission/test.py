import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.transforms import v2

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Define model dictionary
model_dict = {
    'AlexNet': models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
    'VGG16': models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    'ResNet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
}

# Load CIFAR-100 dataset
def load_cifar100(data_dir, batch_size):
    val_transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader

# Top-K error rate calculation
def calculate_top_k_error(output, target, k):
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[:k].reshape(-1).float().sum(0, keepdim=True)

# Test a model
def test_model(model, dataloader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total += targets.size(0)
            top1_correct += calculate_top_k_error(outputs, targets, 1).item()
            top5_correct += calculate_top_k_error(outputs, targets, 5).item()
    
    top1_error = 1 - (top1_correct / total)
    top5_error = 1 - (top5_correct / total)
    return top1_error, top5_error

# Main evaluation script
def evaluate_models(models_dir, data_dir, batch_size, device):
    val_loader = load_cifar100(data_dir, batch_size)
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]

    results = []
    for model_file in model_files:
        print(f"Evaluating model: {model_file}")
        model_path = os.path.join(models_dir, model_file)

        # Identify model architecture from file name
        model_name = model_file.split('_')[0]  # Assuming the model name is the prefix (e.g., AlexNet_*)
        if model_name not in model_dict:
            print(f"Unknown model architecture in {model_file}. Skipping.")
            continue
        
        # Load model and weights
        model = model_dict[model_name]
        if model_name == 'AlexNet':
          model.classifier[6] = nn.Linear(4096, 100)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        # Evaluate model
        top1_error, top5_error = test_model(model, val_loader, device)
        results.append((model_file, top1_error, top5_error))
        print(f"Top-1 Error: {top1_error:.4f}, Top-5 Error: {top5_error:.4f}")
    
    return results

# Parse arguments and run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test models and calculate top-1 and top-5 error rates.")
    parser.add_argument('--models_dir', type=str, required=True, help="Directory containing model weight files.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing CIFAR-100 dataset.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for testing.")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = evaluate_models(args.models_dir, args.data_dir, args.batch_size, DEVICE)

    print("\nFinal Results:")
    for model_file, top1, top5 in results:
        print(f"Model: {model_file}, Top-1 Error: {top1:.4f}, Top-5 Error: {top5:.4f}")
