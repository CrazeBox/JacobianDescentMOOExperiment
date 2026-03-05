"""
Reproduce Jacobian Descent paper results on CIFAR-10
Paper: "Jacobian Descent for Multi-Objective Optimization" (arXiv:2406.16232)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
import random
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# CIFAR-10 normalization constants
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class MultiTaskCIFAR10(Dataset):
    """
    CIFAR-10 dataset for multi-task learning.
    Creates 5 binary classification tasks from 10 classes.
    """
    def __init__(self, root: str, train: bool = True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
        # Map 10 classes to 5 binary tasks
        # Task 0: classes 0,1 vs others
        # Task 1: classes 2,3 vs others
        # etc.
        self.num_tasks = 5
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Create binary labels for each task
        task_labels = []
        for task_id in range(self.num_tasks):
            if label in [2*task_id, 2*task_id+1]:
                task_labels.append(1)
            else:
                task_labels.append(0)
        return img, torch.tensor(task_labels, dtype=torch.long), label


class ResNet18MultiTask(nn.Module):
    """
    ResNet-18 backbone with multiple task-specific heads.
    Following the paper's architecture.
    """
    def __init__(self, num_tasks: int = 5, num_classes_per_task: int = 2):
        super().__init__()
        # Load pretrained ResNet-18
        resnet = torchvision.models.resnet18(pretrained=False)
        
        # Modify first conv for CIFAR-10 (32x32 images)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        
        # Shared backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # Task-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(self.feature_dim, num_classes_per_task) 
            for _ in range(num_tasks)
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        outputs = [head(features) for head in self.heads]
        return outputs


class Aggregator:
    """Base class for multi-objective aggregators."""
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """Simple mean aggregation (baseline)."""
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(gradients).mean(dim=0)


class UPGradAggregator(Aggregator):
    """
    UPGrad aggregator from the paper.
    Projects gradients to resolve conflicts.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def __call__(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        # Stack gradients into matrix
        J = torch.stack(gradients)  # [num_tasks, num_params]
        
        # Compute Gram matrix
        G = torch.matmul(J, J.t())  # [num_tasks, num_tasks]
        
        # Add small epsilon for numerical stability
        G = G + self.epsilon * torch.eye(G.size(0), device=G.device)
        
        # Compute UPGrad weights
        # This is a simplified version - full implementation would use the paper's algorithm
        try:
            # Solve for weights that minimize conflict
            e = torch.ones(G.size(0), device=G.device)
            # Use least squares to find weights
            weights = torch.linalg.lstsq(G, e).solution
            weights = weights / weights.sum()  # Normalize
        except:
            # Fallback to uniform weights if numerical issues
            weights = torch.ones(G.size(0), device=G.device) / G.size(0)
        
        # Weighted combination
        aggregated = torch.matmul(weights, J)
        return aggregated


class JacobianDescentOptimizer:
    """
    Optimizer implementing Jacobian Descent.
    """
    def __init__(self, model: nn.Module, aggregator: Aggregator, lr: float = 0.001):
        self.model = model
        self.aggregator = aggregator
        self.lr = lr
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    def step(self, batch_data, batch_labels):
        """Perform one Jacobian Descent step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch_data)
        
        # Compute per-task losses
        criterion = nn.CrossEntropyLoss()
        losses = []
        for task_id, output in enumerate(outputs):
            loss = criterion(output, batch_labels[:, task_id])
            losses.append(loss)
        
        # Compute per-task gradients
        task_gradients = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            # Collect gradients
            grads = []
            for param in self.model.parameters():
                if param.grad is None:
                    grads.append(torch.zeros_like(param).flatten())
                else:
                    grads.append(param.grad.flatten())
            task_gradients.append(torch.cat(grads))
        
        # Aggregate gradients using the aggregator
        aggregated_gradient = self.aggregator(task_gradients)
        
        # Apply aggregated gradient
        self.optimizer.zero_grad()
        idx = 0
        for param in self.model.parameters():
            num_params = param.numel()
            if param.grad is None:
                param.grad = aggregated_gradient[idx:idx+num_params].view_as(param)
            else:
                param.grad = aggregated_gradient[idx:idx+num_params].view_as(param)
            idx += num_params
        
        self.optimizer.step()
        
        return [loss.item() for loss in losses]


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, device: str):
    """Train for one epoch."""
    model.train()
    total_loss = [0.0] * 5
    num_batches = 0
    
    for batch_data, batch_labels, _ in tqdm(train_loader, desc="Training"):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        losses = optimizer.step(batch_data, batch_labels)
        
        for i, loss in enumerate(losses):
            total_loss[i] += loss
        num_batches += 1
    
    return [loss / num_batches for loss in total_loss]


def evaluate(model: nn.Module, test_loader: DataLoader, device: str):
    """Evaluate model on test set."""
    model.eval()
    correct = [0] * 5
    total = [0] * 5
    
    with torch.no_grad():
        for batch_data, batch_labels, _ in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            
            for task_id, output in enumerate(outputs):
                _, predicted = output.max(1)
                correct[task_id] += (predicted == batch_labels[:, task_id]).sum().item()
                total[task_id] += batch_labels.size(0)
    
    accuracies = [100.0 * c / t for c, t in zip(correct, total)]
    return accuracies


def main():
    # Setup
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    # Create datasets
    data_root = "./data"
    os.makedirs(data_root, exist_ok=True)
    
    train_dataset = MultiTaskCIFAR10(data_root, train=True, transform=transform_train)
    test_dataset = MultiTaskCIFAR10(data_root, train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create model
    model = ResNet18MultiTask(num_tasks=5, num_classes_per_task=2).to(device)
    
    # Test different aggregators
    aggregators = {
        "Mean": MeanAggregator(),
        "UPGrad": UPGradAggregator()
    }
    
    results = {}
    
    for agg_name, aggregator in aggregators.items():
        print(f"\n{'='*50}")
        print(f"Training with {agg_name} aggregator")
        print(f"{'='*50}")
        
        # Reset model for each aggregator
        model = ResNet18MultiTask(num_tasks=5, num_classes_per_task=2).to(device)
        optimizer = JacobianDescentOptimizer(model, aggregator, lr=0.1)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, T_max=200)
        
        # Training loop
        num_epochs = 200
        history = {
            'train_loss': [],
            'test_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_losses = train_epoch(model, train_loader, optimizer, device)
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            # Evaluate
            test_accs = evaluate(model, test_loader, device)
            avg_test_acc = sum(test_accs) / len(test_accs)
            
            history['train_loss'].append(avg_train_loss)
            history['test_acc'].append(avg_test_acc)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Test Acc: {avg_test_acc:.2f}%")
            print(f"Task Accuracies: {[f'{acc:.2f}%' for acc in test_accs]}")
            
            scheduler.step()
        
        results[agg_name] = history
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for agg_name, history in results.items():
        plt.plot(history['train_loss'], label=agg_name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for agg_name, history in results.items():
        plt.plot(history['test_acc'], label=agg_name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('jd_cifar10_results.png', dpi=150)
    plt.show()
    
    # Save results
    import json
    with open('jd_cifar10_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to jd_cifar10_results.png and jd_cifar10_results.json")


if __name__ == "__main__":
    main()
