import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import preprocess_data
from model import WeatherTransformer, get_device
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime

def train_model(X_train, y_train, X_val, y_val, params):
    device = get_device()
    print(f"Using device: {device}")

    horizon = y_train.shape[1]
    num_targets = y_train.shape[2]
    output_size = horizon * num_targets

    model = WeatherTransformer(params['input_size'], params['d_model'], 
                               params['nhead'], params['num_layers'], output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    y_train_flat = y_train.view(y_train.shape[0], -1)
    y_val_flat = y_val.view(y_val.shape[0], -1)

    train_data = TensorDataset(X_train, y_train_flat)
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=False)

    val_data = TensorDataset(X_val, y_val_flat)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

    train_losses = []
    val_losses = []

    start_training_time = time.time()
    print(f"\nStarting training at: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 50)

    for epoch in range(params['epochs']):
        epoch_start_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        model.train()
        batch_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(train_loss)

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_batch_losses.append(loss.item())

        val_loss = sum(val_batch_losses) / len(val_batch_losses)
        val_losses.append(val_loss)

        epoch_duration = time.time() - epoch_start_time
        print(f"[{timestamp}] Epoch [{epoch+1}/{params['epochs']}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} - Duration: {epoch_duration:.2f}s")

    total_time = time.time() - start_training_time
    avg_epoch_time = total_time / params['epochs']
    
    print("-" * 50)
    print(f"Training Complete at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Weather Transformer model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    args = parser.parse_args()

    try:
        X_train, y_train, X_test, y_test, scaler = preprocess_data("weather_data.csv", seq_length=168, horizon=24)

        params = {
            'input_size': X_train.shape[2],
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }

        model, train_losses, val_losses = train_model(X_train, y_train, X_test, y_test, params)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig("loss_plot.png")
        
        torch.save(model.state_dict(), "weather_transformer.pth")
        print("\nModel saved to weather_transformer.pth")

    except FileNotFoundError:
        print("weather_data.csv not found. Run data_fetcher.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")
