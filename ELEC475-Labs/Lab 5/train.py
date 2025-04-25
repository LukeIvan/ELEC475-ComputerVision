import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torchvision.datasets import VOCSegmentation
from torch.nn import CrossEntropyLoss
from torchvision.transforms import InterpolationMode, v2
from model import DMadNet
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


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


def cosine_feature_loss(student_features, teacher_features):
    channel_matcher = torch.nn.Conv2d(
        student_features.shape[1], teacher_features.shape[1], kernel_size=1
    ).to(student_features.device)

    student_features = channel_matcher(student_features)

    target_h = min(student_features.shape[2], teacher_features.shape[2])
    target_w = min(student_features.shape[3], teacher_features.shape[3])

    student_features = F.adaptive_max_pool2d(student_features, (target_h, target_w))
    teacher_features = F.adaptive_max_pool2d(teacher_features, (target_h, target_w))

    student_features = student_features.view(
        student_features.size(0), student_features.size(1), -1
    )
    teacher_features = teacher_features.view(
        teacher_features.size(0), teacher_features.size(1), -1
    )

    student_features = F.normalize(student_features, dim=2)
    teacher_features = F.normalize(teacher_features, dim=2)

    return (1 - F.cosine_similarity(student_features, teacher_features, dim=2)).mean()


def train(epochs=30, distillation_mode="response", **kwargs) -> None:
    model = kwargs["model"]
    teacher_model = kwargs["teacher_model"]
    device = kwargs["device"]
    train_loader = kwargs["train"]
    val_loader = kwargs["test"]
    optimizer = kwargs["optimizer"]
    scheduler = kwargs["scheduler"]
    criterion = kwargs["fn_loss"]

    # 2 for response
    temperature = 2.0
    alpha = 0.5

    losses_train = []
    losses_val = []
    best_val_loss = float("inf")
    early_stopping = EarlyStopping(patience=7)
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad(set_to_none=True)

            student_outputs = model(imgs)

            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)['out']

            ce_loss = criterion(student_outputs, lbls)

            if distillation_mode == "response":
                distill_loss = F.cross_entropy(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    F.softmax(teacher_outputs / temperature, dim=1),
                )
                loss = alpha * distill_loss + (1 - alpha) * ce_loss
            else:
                distill_loss = cosine_feature_loss(student_outputs, teacher_outputs)
                loss = alpha * distill_loss + (1 - alpha) * ce_loss

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        losses_train.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)[0]
                teacher_outputs = teacher_model(imgs)["out"]

                # Calculate both losses as in training
                distillation_loss = F.cross_entropy(
                    F.log_softmax(outputs / temperature, dim=1),
                    F.softmax(teacher_outputs / temperature, dim=1),
                )
                ce_loss = criterion(outputs, lbls)
                loss = alpha * distillation_loss + (1 - alpha) * ce_loss

                val_loss += loss.item()  # Accumulate validation loss

        avg_val_loss = val_loss / len(val_loader)
        losses_val.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                "./training_outputs/{distillation_mode}_model.pth",
            )

        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
            f"Epoch: {epoch}, Training loss: {avg_train_loss:.4f}, "
            f"Validation loss: {avg_val_loss:.4f}, "
        )

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        plt.figure(figsize=(12, 7))
        plt.plot(losses_train, label="Train Loss")
        plt.plot(losses_val, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.savefig("./training_outputs/loss_plot_{distillation_mode}.png")
        plt.close()

    end = time.time()
    elapsed_time = (end - start) / 60
    print(f"Training completed in {elapsed_time:.2f} minutes")


def calculate_class_weights(dataset):
    class_counts = torch.zeros(21)
    for _, mask in dataset:
        unique_labels, counts = torch.unique(mask, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label != 255:
                class_counts[label] += count

    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights) 
    weights[0] = 0.5
    return weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model_path = "./training_outputs/base_model.pth"
    teacher_model = fcn_resnet50().to(device)

    # config = DMadNetConfig(
    #     in_channels=3, num_classes=21, initial_filters=32, bilinear=True
    # )

    student_model = DMadNet(n=21).to(device)
    student_model.load_state_dict(torch.load(student_model_path)["model_state_dict"])

    teacher_model.eval()

    # Data augmentation and transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
        ]
    )

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
        train_dataset, batch_size=21, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=21, shuffle=False, num_workers=4, pin_memory=True
    )

    optimizer = optim.Adam(student_model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    class_weights = calculate_class_weights(train_dataset)

    criterion = CrossEntropyLoss(
        weight=class_weights, reduction="mean", ignore_index=255
    ).to(device)

    train(
        distillation_mode="response",
        optimizer=optimizer,
        model=student_model,
        teacher_model=teacher_model,
        fn_loss=criterion,
        train=train_loader,
        test=val_loader,
        scheduler=scheduler,
        device=device,
    )


if __name__ == "__main__":
    main()
