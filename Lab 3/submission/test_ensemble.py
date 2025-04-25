import os
import torch
import torch.nn as nn

from torchvision import models, datasets, transforms
from torchvision.transforms import v2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

model_dict = {
    "AlexNet": models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
    "VGG16": models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    "ResNet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT, ),
}



def load_weights(models_dir, model_names, device):
    models = {}
    for model_name in model_names:
        model_files = [
            f for f in os.listdir(models_dir) 
            if f.startswith(model_name) and (f.endswith("_final.pth") or f.endswith("_epoch5.pth"))
        ]
        model = model_dict[model_name].to(device)
        if model_name == 'VGG16': model.classifier[6] = nn.Linear(4096, 100)
        if model_name == 'ResNet18': model.fc = nn.Linear(512, 100)
        if model_name == 'AlexNet': model.classifier[6] = nn.Linear(4096, 100)
        for model_file in model_files:
            weights_path = os.path.join(models_dir, model_file)
            state_dict = torch.load(weights_path)
            if model_name == 'VGG16':
                if 'classifier.6.weight' in state_dict: state_dict['classifier.6.weight'] = state_dict['classifier.6.weight'][:100]
                if 'classifier.6.bias' in state_dict: state_dict['classifier.6.bias'] = state_dict['classifier.6.bias'][:100]
            if model_name == 'ResNet18':
                if 'fc.weight' in state_dict: state_dict['fc.weight'] = state_dict['fc.weight'][:100, :]
                if 'fc.bias' in state_dict: state_dict['fc.bias'] = state_dict['fc.bias'][:100]
            model.load_state_dict(state_dict, strict=False)
        
        models[model_name] = model
    return models


def load_cifar100(data_dir, batch_size):
    val_transform = transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet normalization
        ]
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return val_loader


def max_probability_ensemble(outputs):
    return torch.argmax(torch.max(torch.stack(outputs, dim=0), dim=0).values, dim=1)


def probability_averaging_ensemble(outputs):
    return torch.argmax(torch.mean(torch.stack(outputs, dim=0), dim=0), dim=1)


def majority_voting_ensemble(outputs):
    preds = torch.stack([torch.argmax(o, dim=1) for o in outputs], dim=0)
    return torch.mode(preds, dim=0).values

def test_ensemble(models, dataloader, device, method):
    for model in models.values():
        model.to(device)
        model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing Ensemble"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = [F.softmax(model(inputs), dim=1) for model in models.values()]
            predictions = method(outputs)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return 1 - correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ensemble methods on CIFAR-100.")
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory containing model weight files.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing CIFAR-100 dataset.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for testing."
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader = load_cifar100(args.data_dir, args.batch_size)

    model_names = ["AlexNet", "VGG16", "ResNet18"]
    models = load_weights(args.models_dir, model_names, device)

    methods = {
        "Max Probability": max_probability_ensemble,
        "Probability Averaging": probability_averaging_ensemble,
        "Majority Voting": majority_voting_ensemble,
    }

    for method_name, method in methods.items():
        error = test_ensemble(models, val_loader, device, method)
        print(f"{method_name} Error Rate: {error:.4f}")
