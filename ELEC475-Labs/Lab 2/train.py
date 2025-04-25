import argparse
import torch
import torch.nn as nn
from dataset import PetNosesDataset
from model import CNN
from plot import Plot
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def get_transform(args):
    transform_list = [
        v2.Resize((227, 227)),
        v2.ToImage(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if args.color_jitter:
        transform_list.insert(
            1,
            v2.ColorJitter(
                brightness=(0.2, 1.5), contrast=(0.2, 1.5), saturation=(0.2, 1.5)
            ),
        )

    if args.gaussian_blur:
        transform_list.insert(1, v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)))

    if args.random_erasing:
        transform_list.insert(
            1, v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        )

    return v2.Compose(transform_list)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.apply(initialize_weights)

    loss_function = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Set up plot and transform
    loss_plot = Plot(
        # log_scale=True,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    transform = get_transform(args)

    # Datasets and DataLoaders
    train_dataset = PetNosesDataset(
        txt_file=args.train_txt, root_dir=args.data_path, transform=transform
    )
    val_dataset = PetNosesDataset(
        txt_file=args.val_txt, root_dir=args.data_path, transform=transform
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_loss_values, val_loss_values = [], []
    best_val_loss_avg = float("inf")
    no_improvement_epochs = 0
    val_loss_history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for images, nose_coords in train_loader:
            images, nose_coords = images.to(device), nose_coords.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = loss_function(outputs, nose_coords)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, norm_type=2
            )
            optimizer.step()
            train_loss_sum += train_loss.item()

        train_loss_avg = train_loss_sum / len(train_loader)
        train_loss_values.append(train_loss_avg)

        # Validation
        model.eval()
        val_loss_sum, distances = 0.0, []
        with torch.no_grad():
            for images, nose_coords in val_loader:
                images, nose_coords = images.to(device), nose_coords.to(device)
                outputs = model(images)
                val_loss = loss_function(outputs, nose_coords)
                val_loss_sum += val_loss.item()
                distances.extend(
                    torch.sqrt(torch.sum((outputs - nose_coords) ** 2, dim=1))
                    .cpu()
                    .numpy()
                )

        val_loss_avg = val_loss_sum / len(val_loader)
        val_loss_values.append(val_loss_avg)
        val_loss_history.append(val_loss_avg)

        # Calculate the average validation loss over the past `patience` epochs
        if len(val_loss_history) > args.patience:
            val_loss_avg = sum(val_loss_history[-args.patience :]) / args.patience
        else:
            val_loss_avg = sum(val_loss_history) / len(val_loss_history)

        # Update the learning rate using the scheduler
        scheduler.step(val_loss_avg)
        print(
            f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss_avg:.4f}, Validation Loss: {val_loss_avg:.4f}"
        )
        loss_plot.update(epoch + 1, train_loss=train_loss_avg, val_loss=val_loss_avg)

        # Early stopping
        if val_loss_avg < best_val_loss_avg:
            best_val_loss_avg = val_loss_avg
            no_improvement_epochs = 0
            torch.save(model.state_dict(), "snoutnet.pth")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= args.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    loss_plot.finalize()
    return train_loss_avg, val_loss_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SnoutNet")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Dataset directory"
    )
    parser.add_argument("--train_txt", type=str, help="Path to train txt file")
    parser.add_argument(
        "--val_txt", type=str, help="Path to validation (test) txt file"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--color_jitter", action="store_true", help="Apply color jitter augmentation"
    )
    parser.add_argument(
        "--gaussian_blur", action="store_true", help="Apply gaussian blur augmentation"
    )
    parser.add_argument(
        "--random_erasing",
        action="store_true",
        help="Apply random erasing augmentation",
    )
    args = parser.parse_args()

    main(args)
