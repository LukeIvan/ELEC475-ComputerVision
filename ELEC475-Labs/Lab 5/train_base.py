import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torchvision.datasets import VOCSegmentation
from torch.nn import CrossEntropyLoss
from torchvision.transforms import InterpolationMode, v2
from model import DMadNet
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(epochs=30, **kwargs) -> None:
    model = kwargs["model"]
    device = kwargs["device"]
    train_loader = kwargs["train"]
    val_loader = kwargs["test"]
    optimizer = kwargs["optimizer"]
    scheduler = kwargs["scheduler"]
    criterion = kwargs["fn_loss"]

    losses_train = []
    losses_val = []
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=5)
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        losses_train.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()


        avg_val_loss = val_loss / len(val_loader)
        losses_val.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, './training_outputs/base_model.pth')

        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
            f"Epoch: {epoch}, Training loss: {avg_train_loss:.4f}, "
            f"Validation loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Plot loss curves
        plt.figure(figsize=(12, 7))
        plt.plot(losses_train, label="train")
        plt.plot(losses_val, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title(f"Training and Validation Loss (Epoch {epoch})")
        plt.savefig(f"./training_outputs/loss_plot_base.png")
        plt.close()

    end = time.time()
    elapsed_time = (end - start) / 60
    print(f"Training completed in {elapsed_time:.2f} minutes")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DMadNet(n=21).to(device)


    # Data augmentation and transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
    ])

    # Dataset initialization with augmentation
    train_dataset = VOCSegmentation(
        "./data",
        image_set="train",
        transform=train_transform,
        target_transform=mask_transform,
        download=False,
    )

    val_dataset = VOCSegmentation(
        "./data",
        image_set="val",
        transform=val_transform,
        target_transform=mask_transform,
        download=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    class_weights = torch.ones(21).to(device)
    class_weights[0] = 0.5
    criterion = CrossEntropyLoss(
        weight=class_weights, reduction="mean", ignore_index=255
    ).to(device)

    train(
        optimizer=optimizer,
        model=model,
        fn_loss=criterion,
        train=train_loader,
        test=val_loader,
        scheduler=scheduler,
        device=device,
    )

if __name__ == "__main__":
    main()
