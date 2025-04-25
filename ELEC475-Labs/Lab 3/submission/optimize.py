import optuna
import torch
import os
import torch.nn as nn
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.transforms import v2
from argparse import Namespace
from loguru import logger

from plot import Plot  

os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "training_log_{time:YYYY-MM-DD_HH-mm-ss}.log")
logger.add(log_file, format="{time} {level} {message}")



# Redirect Optuna logging to Loguru
def redirect_optuna_logging():
    import logging
    from optuna import logging as optuna_logging

    class LoguruHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            logger.log(record.levelname, log_entry)

    loguru_handler = LoguruHandler()
    loguru_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    loguru_handler.setFormatter(formatter)

    optuna_logger = optuna_logging.get_logger("optuna")
    optuna_logger.handlers.clear()
    optuna_logger.addHandler(loguru_handler)

redirect_optuna_logging()

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


def train(args, model, criterion, optimizer, scheduler, train_loader, val_loader, trial, device):
    checkpoint_filename = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_wd{args.weight_decay}"

    best_loss = float('inf')
    patience = args.patience

    # plotter = Plot(model_name=model_name, epochs=args.num_epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, batch_size=args.batch_size)

    logger.info(f"Training {args.model_name} with hyperparameters: lr={args.learning_rate}, batch_size={args.batch_size}, weight_decay={args.weight_decay}")

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
                    # plotter.update(epoch, train_loss.cpu().item(), avg_val_loss)
                    logger.info(f"Epoch [{epoch}/{args.num_epochs}] - Train Loss: {train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")
                    logger.warning("Early stopping triggered.")
                    break
        

        # plotter.update(epoch, train_loss.cpu().item(), avg_val_loss)
        scheduler.step(avg_val_loss)
        logger.info(f"Epoch [{epoch}/{args.num_epochs}] - Train Loss: {train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

        if epoch == 5:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{checkpoint_filename}_epoch5.pth'))

        trial.report(avg_val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{checkpoint_filename}_final.pth'))
    logger.success(f"Saved {args.model_name} parameters after full convergence.")
    # plotter.finalize()
    return best_loss

def optimize(trial):
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3)

    model_name = "VGG"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
    # model.classifier[6] = nn.Linear(4096, 100)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    args = Namespace(
        batch_size=batch_size,
        num_epochs=30,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=5,
        model_name=model_name,
        data_dir='./dataset',
        save_dir='./model_checkpoints_optuna'
    )

    model = model.to(device)
    train_loader, val_loader = load_cifar100(args.data_dir, args.batch_size)

    try:
        best_val_loss = train(args, model, criterion, optimizer, scheduler, train_loader, val_loader, trial, device)
        return best_val_loss
    except optuna.exceptions.TrialPruned:
        logger.warning("Trial was pruned.")
        raise
    except Exception as e:
        logger.error(f"Error occurred  during optimization: {e}")
        return float('inf')


if __name__ == "__main__":
    torch.cuda.empty_cache()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study_path = 'sqlite:///optuna_study_new.db'
    study = optuna.create_study(storage=study_path, direction='minimize', study_name='vgg', load_if_exists=True, pruner=pruner)

    try:
        study.optimize(optimize, n_trials=50, n_jobs=1)
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}")

    logger.success(f"Best hyperparameters: {study.best_params}")
    logger.success(f"Best validation loss: {study.best_value}")
