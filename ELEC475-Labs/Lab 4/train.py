import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torchvision.datasets import VOCSegmentation
from torch.nn import CrossEntropyLoss
from torchvision.transforms import InterpolationMode, v2
from model2 import DMadNet
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        # Training loop
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, lbls in train_bar:
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        losses_train.append(avg_train_loss)

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for imgs, lbls in val_bar:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                val_bar.set_postfix({"loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)
        losses_val.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
            f"Epoch: {epoch}, Training loss: {avg_train_loss:.4f}, "
            f"Validation loss: {avg_val_loss:.4f}"
        )

        # Save weights
        torch.save(model.state_dict(), f"./training_outputs/weights_epoch_{epoch}.pth")

        # Plot and save loss curve
        plt.figure(figsize=(12, 7))
        plt.plot(losses_train, label="train")
        plt.plot(losses_val, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title(f"Training and Validation Loss (Epoch {epoch})")
        plt.savefig(f"./training_outputs/loss_plot_epoch_{epoch}.png")
        plt.close()

    end = time.time()
    elapsed_time = (end - start) / 60
    print(f"Training completed in {elapsed_time:.2f} minutes")

    # Save training time
    with open("./training_outputs/training_time.txt", "w") as f:
        f.write(f"Training time: {elapsed_time:.2f} minutes")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMadNet(n=21).to(device)

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    mask_transformations = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
    ])

    # Dataset initialization
    train_dataset = VOCSegmentation(
        "./data",
        image_set="train",
        transform=transformations,
        target_transform=mask_transformations,
        download=False,
    )

    val_dataset = VOCSegmentation(
        "./data",
        image_set="val",
        transform=transformations,
        target_transform=mask_transformations,
        download=False,
)


    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
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