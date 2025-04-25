import optuna
import torch
import argparse
import logging
from train import main
from typing import Any, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Configuration for dataset paths
DATA_CONFIG = {
    "data_path": "dataset/oxford-iiit-pet-noses/images-original/images",
    "train_txt": "dataset/oxford-iiit-pet-noses/train_noses.txt",
    "val_txt": "dataset/oxford-iiit-pet-noses/test_noses.txt",
}

def create_args(trial: optuna.Trial) -> argparse.Namespace:
    """Create argument namespace for training."""
    return argparse.Namespace(
        data_path=DATA_CONFIG["data_path"],
        train_txt=DATA_CONFIG["train_txt"],
        val_txt=DATA_CONFIG["val_txt"],
        batch_size=trial.suggest_int("batch_size", 16, 64, step=16),
        epochs=trial.suggest_int("epochs", 10, 100, step=10),
        patience=5,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        color_jitter=True,
        gaussian_blur=True,
        random_erasing=False,
    )

def log_trial_info(trial: optuna.Trial, args: argparse.Namespace) -> None:
    """Log the current trial's hyperparameters."""
    logging.info(f"Trial {trial.number}: "
                 f"Learning Rate: {args.learning_rate:.6f}, "
                 f"Weight Decay: {args.weight_decay:.6f}, "
                 f"Batch Size: {args.batch_size}, "
                 f"Epochs: {args.epochs}")

def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization."""
    args = create_args(trial)
    log_trial_info(trial, args)
    
    try:
        train_loss, val_loss = main(args)
        logging.info(f"Trial {trial.number} completed with validation loss: {val_loss:.6f}")
        return val_loss
    except KeyboardInterrupt:
        logging.warning("Trial interrupted. Proceeding to the next trial.")
        return float("inf")  # Return a high loss to skip this trial
    except Exception as e:
        logging.error(f"Trial failed due to an exception: {e}")
        return float("inf")  # Return a high loss to skip this trial


def hyperparameter_optimization() -> None:
    """Run hyperparameter optimization using Optuna."""
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

    # Log the best hyperparameters and their corresponding loss
    logging.info("Best hyperparameters: ")
    logging.info(study.best_params)
    logging.info(f"Best validation loss: {study.best_value:.6f}")

if __name__ == "__main__":
    hyperparameter_optimization()
