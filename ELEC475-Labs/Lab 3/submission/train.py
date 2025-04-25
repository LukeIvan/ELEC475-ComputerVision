import argparse
import torch
import torch.nn as nn
import os

from torchvision import models, datasets, transforms
from torchvision.transforms import v2
from plot import Plot  
from loguru import logger

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train ensemble models on CIFAR100 dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 regularization).")
    parser.add_argument("--save_dir", type=str, default="./model_checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Directory for CIFAR100 dataset.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--model", type=str, choices=["AlexNet", "VGG16", "ResNet18"], required=True, help="Model to train.")

    return parser.parse_args()

# Training function
def train(args, model, criterion, optimizer, scheduler):
    logger.info(f"Training {model_name} with hyperparameters: lr={args.learning_rate}, batch_size={args.batch_size}, weight_decay={args.weight_decay}")

    checkpoint_filename = f"{args.model}_lr{args.learning_rate:.5f}_bs{args.batch_size:.5f}_wd{args.weight_decay:.5f}"

    best_loss = float('inf')
    patience = args.patience

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    plotter = Plot(model_name=model_name, epochs=args.num_epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, batch_size=args.batch_size)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward Pass
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)

            # Backwards Pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
        
            avg_val_loss = running_val_loss / len(val_loader)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience = args.patience
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'{checkpoint_filename}_best.pth'))
            else:
                patience -= 1
                logger.warning(f"Validation loss did not improve. Patience counter: {patience}")
                if patience == 0: 
                    plotter.update(epoch, train_loss.cpu().item(), avg_val_loss)
                    logger.info(f"Epoch [{epoch}/{args.num_epochs}] - Train Loss: {train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")
                    logger.warning("Early stopping triggered.")
                    break
        

        plotter.update(epoch, train_loss.cpu().item(), avg_val_loss)
        scheduler.step(avg_val_loss)
        logger.info(f"Epoch [{epoch}/{args.num_epochs}] - Train Loss: {train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

        if epoch == 5:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{checkpoint_filename}_epoch5.pth'))
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{checkpoint_filename}_final.pth'))
    logger.success(f"Saved {model_name} parameters after full convergence.")
    plotter.finalize()
    return best_loss

# Data Loader
def load_cifar100(data_dir, batch_size):
    train_transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CFAR
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
    ])

    val_transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224), antialias=True),  # Deterministic resize for consistency
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last = True
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last = False
    )
    return train_loader, val_loader


if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader = load_cifar100(args.data_dir, args.batch_size)

    model_dict = {
        'AlexNet': models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
        'VGG16': models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        'ResNet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    }

    model_name = args.model
    model = model_dict[model_name]

    if model_name == 'AlexNet':
        model.classifier[6] = nn.Linear(4096, 100)
    

    logger.info(f"Training {model_name}...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    train(args, model, criterion, optimizer, scheduler)
