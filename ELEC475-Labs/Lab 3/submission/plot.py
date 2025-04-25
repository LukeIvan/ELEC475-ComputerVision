import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
import os

class Plot:
    def __init__(self, log_scale=False, **kwargs):
        plt.style.use("seaborn-v0_8-whitegrid")
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        plt.rcParams.update({
            "font.size": 14,
            "font.family": "sans-serif",
            "font.weight": "bold",
        })
        
        self.train_loss_values = []
        self.val_loss_values = []
        self.epochs = []
        self.log_scale = log_scale
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.train_line, = self.ax.plot([], [], color=self.colors[0], label="Training Loss", linewidth=2)
        self.val_line, = self.ax.plot([], [], color=self.colors[1], label="Validation Loss", linewidth=2)
        
        self.hyperparams = kwargs
        self.configure_axes()
        
        plt.ion()
        plt.show()

    def configure_axes(self):
        self.ax.set_xlabel("Epochs", fontsize=14, fontweight="bold")
        self.ax.set_ylabel("Loss", fontsize=14, fontweight="bold")
        self.ax.set_title(f"Training and Validation Loss: {self.hyperparams['model_name']}", fontsize=16, fontweight="bold")
        
        self.ax.legend(loc="upper right", frameon=False, fontsize=12)
        self.ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
        self.ax.tick_params(axis="both", which="major", labelsize=12)
        
        # Configure y-axis for symlog if needed
        if self.log_scale:
            self.ax.set_yscale("symlog", linthresh=0.1)
        
        # Format y-axis labels as plain numbers
        scalar_formatter = ScalarFormatter()
        scalar_formatter.set_scientific(False)
        scalar_formatter.set_useOffset(False)
        self.ax.yaxis.set_major_formatter(scalar_formatter)
        
        plt.tight_layout()

    def update(self, epoch, train_loss, val_loss):
        self.train_loss_values.append(train_loss)
        self.val_loss_values.append(val_loss)
        self.epochs.append(epoch)
           
        # Update y-axis scale and dynamic limits
        self.ax.set_ylim(0, (max(self.train_loss_values + self.val_loss_values)) * 1.1)
        self.ax.set_xlim(1, (max(self.epochs)))


        # Refresh data in plot lines
        self.train_line.set_data(self.epochs, self.train_loss_values)
        self.val_line.set_data(self.epochs, self.val_loss_values)
        
        # Update the plot with latest epoch details
        self.ax.legend(
            [f"Training Loss: {train_loss:.2f}", f"Validation Loss: {val_loss:.2f}"],
            loc="upper right",
            frameon=False,
            fontsize=12,
        )
        
        # Update x-ticks
        self.ax.set_xticks(range(1, len(self.epochs) + 1))
        plt.draw()
        plt.pause(0.01)

    def finalize(self):
        plt.ioff()
        
        results_folder = "results"
        os.makedirs(results_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hyperparams_str = "_".join(f"{k}={v}" for k, v in self.hyperparams.items())
        filename = f"{results_folder}/{timestamp}_loss_plot_{hyperparams_str}.png"
        
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close(self.fig)
