import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torchvision.datasets import VOCSegmentation
from torch.nn import CrossEntropyLoss, functional
from torchvision.transforms import InterpolationMode
from model import DMadNet 
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train DMadNet on VOC12 Dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization).")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for hard/soft loss balance")

    return parser.parse_args()
"""
temp = 4.0 per hinton
alpha = 0.5 per hinton
b = 1-a
"""
def distillation_loss(student_outputs, teacher_outputs, labels, temperature=6.0, alpha=0.7):
    # Hard loss
    hard_loss = functional.cross_entropy(student_outputs, labels, reduction='mean', ignore_index=255)
    
    # Soft loss
    soft_loss = functional.cross_entropy(
        functional.log_softmax(student_outputs / temperature, dim=1),
        functional.softmax(teacher_outputs / temperature, dim=1),
    )
    
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return total_loss

def feature_distillation(student_outputs, teacher_outputs, labels, alpha = 0.2, temperature = 2):
    ce_loss = CrossEntropyLoss(ignore_index=255)(student_outputs, labels)
    t_features = teacher_outputs
    s_features = student_outputs
    if s_features.shape != t_features.shape:
        t_features = torch.nn.functional.interpolate(
        t_features, size=s_features.shape[2:], mode='bilinear', align_corners=False
    )

    distillation_loss = torch.nn.functional.mse_loss(s_features, t_features)

    return alpha * ce_loss + (1 - alpha) * distillation_loss

def train(args, **kwargs) -> None:
    epochs = args.num_epochs
    print("Training Parameters")
    for kwarg in kwargs:
        print(kwarg, "=", kwargs[kwarg])
    student = kwargs['student']
    teacher = kwargs['teacher']
    device = kwargs['device']
    optimizer = kwargs['optimizer']
    criterion = kwargs['fn_loss']
    
    teacher.eval()
    student.train()

    losses_train = []
    losses_val = []
    start = time.time()

    for epoch in range(1, epochs+1):
        student.train()
        print("Epoch:", epoch)
        loss_train = 0.0
        for data in kwargs['train_loader']:
            imgs, lbls = data
            imgs = imgs.to(device=device)
            lbls = lbls.to(device=device)

            with torch.no_grad():
                teacher_outputs = teacher(imgs)['out']  # FCN-ResNet50 returns a dict
                teacher_outputs = functional.interpolate(
                    teacher_outputs,
                    size=lbls.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            optimizer.zero_grad(set_to_none=True)
            student_outputs = student(imgs)
            if student_outputs.shape[-2:] != lbls.shape[-2:]:
                student_outputs = functional.interpolate(
                    student_outputs,
                    size=lbls.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            loss = feature_distillation(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                labels=lbls,
                temperature=args.temperature,
                alpha=args.alpha
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            loss_train += loss.item()
            
            # Inside training loop, add these debug prints:
            # print(f"Student output range: {student_outputs.min():.2f} to {student_outputs.max():.2f}")
            # print(f"Teacher output range: {teacher_outputs.min():.2f} to {teacher_outputs.max():.2f}")

        losses_train.append(loss_train/len(kwargs['train_loader']))

        student.eval()
        loss_val = 0.0
        
        with torch.no_grad():
            for imgs, lbls in kwargs["val_loader"]:
                imgs = imgs.to(device=device)
                lbls = lbls.to(device=device)
                
                teacher_outputs = teacher(imgs)['out']
                teacher_outputs = functional.softmax(teacher_outputs, dim=1)
                student_outputs = student(imgs)
                
                loss = feature_distillation(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    labels=lbls,
                    temperature=args.temperature,
                    alpha=args.alpha
                )
                loss_val += loss.item()

        kwargs['scheduler'].step(loss_val)
        losses_val.append(loss_val / len(kwargs["val_loader"]))
        
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "Epoch:", epoch, "Training loss:", loss_train/len(kwargs['train_loader']), "Validation loss:", loss_val / len(kwargs['val_loader']))

        filename = "./training_outputs/weights.pth"
        print("Saving Weights to", filename)
        torch.save(student.state_dict(), filename)

        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        filename = "./training_outputs/loss_plot.png"
        print('Saving to Loss Plot at', filename)
        plt.savefig(filename)
    end = time.time()
    elapsed_time = (end - start) / 60
    print("Training completed in", round(elapsed_time, 2), "minutes")
    time_filename = './training_outputs/training_time.txt'
    

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMadNet(n=21).to(device)
    teacher = fcn_resnet50(pretrained=True)
    student = model.to(device)
    teacher = teacher.to(device)

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
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    class_weights = torch.ones(21).to(device)
    class_weights[0] = 0.5
    criterion = CrossEntropyLoss(
        weight=class_weights, reduction="mean", ignore_index=255
    ).to(device)
    train(
            args=args,
            optimizer=optimizer,
            student=student,
            teacher=teacher,
            fn_loss=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            device=device
            )

if __name__ == "__main__":
    main()