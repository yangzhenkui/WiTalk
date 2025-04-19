import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/0225')

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

class AverageMeter:
    """Tracks and computes the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value and count."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

class Trainer:
    """Handles model training and evaluation with mixed precision and TensorBoard logging."""
    def __init__(self, model, optimizer, scheduler, device="cuda:0", writer=writer):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.writer = writer
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self, data_loader, labels, epoch):
        """Trains the model for one epoch and logs metrics to TensorBoard."""
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                logits = self.model(inputs, labels)
                loss = self.loss_fn(logits, targets)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # Update meters
            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            accuracy = (torch.argmax(logits, dim=1) == targets).float().mean().item()
            acc_meter.update(accuracy, batch_size)

            # Log to TensorBoard
            global_step = epoch * len(data_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Accuracy', accuracy, global_step)

        return loss_meter.avg, acc_meter.avg

    def eval_epoch(self, data_loader, labels, epoch):
        """Evaluates the model for one epoch and logs metrics to TensorBoard."""
        self.model.eval()
        acc_meter = AverageMeter()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs, labels)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == targets).float().mean().item()
                acc_meter.update(accuracy, inputs.size(0))

                # Log to TensorBoard
                global_step = epoch * len(data_loader) + batch_idx
                self.writer.add_scalar('Eval/Accuracy', accuracy, global_step)

        return acc_meter.avg