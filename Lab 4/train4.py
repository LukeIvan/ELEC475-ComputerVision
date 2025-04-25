import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import argparse
from torchvision.datasets import VOCSegmentation
from torch.nn import CrossEntropyLoss
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import fcn_resnet50
from model2 import DMadNet
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

def response_distillation(s_logits, t_logits, label, alpha=0.5, temperature=2):
    ce_loss = CrossEntropyLoss(ignore_index=255)(s_logits, label)
    t_soft = torch.nn.functional.softmax(t_logits / temperature, dim=1)
    s_soft = torch.nn.functional.log_softmax(s_logits / temperature, dim=1)
    distillation_loss = torch.nn.functional.kl_div(
        s_soft, t_soft, reduction='batchmean'
    ) * (temperature ** 2)
    return alpha * ce_loss + (1 - alpha) * distillation_loss

def feature_distillation(s_logits, t_logits, label, alpha=0.5, temperature=2):
    ce_loss = CrossEntropyLoss(ignore_index=255)(s_logits, label)
    if s_logits.shape != t_logits.shape:
        t_logits = torch.nn.functional.interpolate(
            t_logits, size=s_logits.shape[2:], 
            mode='bilinear', align_corners=False
        )
    distillation_loss = torch.nn.functional.mse_loss(s_logits, t_logits)
    return alpha * ce_loss + (1 - alpha) * distillation_loss

def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return 100 * (preds == labels).sum().item() / torch.numel(labels)

def train(student, teacher, train_loader, test_loader, optimizer, 
          scheduler, device, loss_fn, epochs=30):
    teacher.eval()
    losses_train, losses_val = [], []
    output_dir = f'./output/{loss_fn.__name__}' if not isinstance(
        loss_fn, CrossEntropyLoss) else './output/none'
    os.makedirs(output_dir, exist_ok=True)
    
    start = time.time()
    for epoch in range(1, epochs + 1):
        # Training phase
        student.train()
        train_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            s_logits = student(imgs)
            t_logits = teacher(imgs)['out']
            
            loss = (loss_fn(s_logits, lbls) if isinstance(loss_fn, CrossEntropyLoss)
                   else loss_fn(s_logits, t_logits, lbls))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        losses_train.append(avg_train_loss)
        
        # Validation phase
        student.eval()
        val_loss = teacher_acc = student_acc = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                s_logits = student(imgs)
                t_logits = teacher(imgs)['out']
                
                loss = (loss_fn(s_logits, lbls) if isinstance(loss_fn, CrossEntropyLoss)
                       else loss_fn(s_logits, t_logits, lbls))
                
                val_loss += loss.item()
                teacher_acc += calculate_accuracy(t_logits, lbls)
                student_acc += calculate_accuracy(s_logits, lbls)
        
        avg_val_loss = val_loss / len(test_loader)
        avg_teacher_acc = teacher_acc / len(test_loader)
        avg_student_acc = student_acc / len(test_loader)
        
        scheduler.step(avg_val_loss)
        losses_val.append(avg_val_loss)
        
        # Save progress
        torch.save(student.state_dict(), f"{output_dir}/weights.pth")
        with open(f"{output_dir}/accuracy.txt", "a") as f:
            f.write(f"Epoch {epoch}: Teacher: {avg_teacher_acc:.2f}%, Student: {avg_student_acc:.2f}%\n")
        
        # Plot losses
        plt.figure(figsize=(12, 7))
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{output_dir}/loss_plot.png")
        plt.close()
        
    print(f"Training completed in {(time.time() - start) / 60:.2f} minutes")

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('-e', type=int, default=30, help='Number of epochs')
    parser.add_argument('-b', type=int, default=16, help='Batch size')
    parser.add_argument('--loss', choices=['response', 'feature', 'none'], 
                       default='response', help='Distillation method')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = DMadNet(n=21).to(device)
    teacher = fcn_resnet50(pretrained=True).to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
    ])

    train_dataset = VOCSegmentation('./data', image_set='train', download=False,
                                  transform=transform, target_transform=mask_transform)
    test_dataset = VOCSegmentation('./data', image_set='val', download=False,
                                 transform=transform, target_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=True,
                           num_workers=4, pin_memory=True)

    optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = {'response': response_distillation,
               'feature': feature_distillation,
               'none': CrossEntropyLoss(ignore_index=255)}[args.loss]

    train(student, teacher, train_loader, test_loader, optimizer,
          scheduler, device, loss_fn, args.e)

if __name__ == "__main__":
    main()
