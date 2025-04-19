import os
import torch
import random
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Import custom utilities and classes
from data_util import NpyDataset, select_gpu, get_motion_desc_list
from XRF55.CombinedModel import CombinedModel
from XRF55.resnet1d_wifi import resnet18
from Trainer import Trainer

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load datasets and create data loaders
def load_data(batch_size):
    train_data = "/home/dataset/XRFDataset/new_data/train_data/WiFi"
    test_data = "/home/dataset/XRFDataset/new_data/test_data/WiFi"
    
    train_dataset = NpyDataset(train_data)
    test_dataset = NpyDataset(test_data)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

# Main training function
def main():
    # Load data
    train_loader, test_loader = load_data(batch_size)
    labels = None
    
    # Create model save directory
    os.makedirs("./model", exist_ok=True)
    
    # Initialize model (with text fusion)
    csi_model = resnet18()
    model = CombinedModel(
        csi_model,
        model_key=model_key,
        embed_type=embed_type,
        device=device,
        is_text=True
    ).to(device)
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.5)
    
    # Initialize trainer
    trainer = Trainer(model, optimizer, scheduler, device)
    
    # Training loop
    num_epochs = 200
    best_acc = 0.0
    model_path = f"./model/WiFi_best_{model_key}_{embed_type}.pth"
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Train and evaluate
        train_loss, train_acc = trainer.train_epoch(train_loader, labels, epoch)
        eval_acc = trainer.eval_epoch(test_loader, labels, epoch)
        scheduler.step()
        
        # Log results
        log_msg = (f"Epoch [{epoch}/{num_epochs}] "
                  f"=> train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, eval_acc={eval_acc:.4f}")
        print(log_msg)
        
        # Save best model
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), model_path)
    
    print(f"Best accuracy ({model_key}_{embed_type}): {best_acc:.4f}")

if __name__ == "__main__":
    # Set random seed and select device
    set_seed(3407)
    device, _ = select_gpu()
    
    # Record start time
    start_time = time.time()
    
    # Hyperparameters
    lr = 0.0001
    batch_size = 64
    model_key = "clip-vit-large-patch14"
    embed_type = "simple"
    
    # Run training
    main()
    
    # Calculate and display execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"Execution time: {hours:02d}h{minutes:02d}min")