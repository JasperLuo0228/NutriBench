import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        epoch = df['epoch'].tolist()
        val_acc = df['val_acc'].tolist()
        val_mae = df['val_mae'].tolist()
        val_rmse = df['val_rmse'].tolist()
        train_acc = df['train_acc'].tolist()
        train_rmse = df['train_rmse'].tolist()
        
        return epoch, val_acc, val_mae, val_rmse, train_acc, train_rmse

    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

def plot(file_path):
    epoch, val_acc, val_mae, val_rmse, train_acc, train_rmse = load_data(file_path)

    plt.figure(figsize=(10, 5))
    plt.plot(epoch, val_acc, label="Val Acc", linewidth=1)
    plt.plot(epoch, train_acc, label="Train Acc", linewidth=1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/accuracy_epochs.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch, train_rmse, label="Train RMSE", linewidth=1)
    plt.plot(epoch, val_rmse, label="Val RMSE", linewidth=1)

    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("RMSE over Epochs")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/rmse_epochs.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch, val_mae, label="Val MAE", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE")
    plt.title("Val MAE over Epochs")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/mae_epochs.png")
    plt.close()

if __name__ == "__main__":
    log_path = "output/log.csv"
    plot(log_path)
    print("Plots saved successfully.")