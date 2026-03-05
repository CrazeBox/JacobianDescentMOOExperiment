"""
Main script to run Jacobian Descent experiments on CIFAR-10.
Reproduces results from the paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from tqdm import tqdm
import yaml

from aggregators import get_aggregator


# CIFAR-10 normalization constants
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def normalize_config(raw_config):
    """Normalize nested/flat config into a flat runtime config."""
    experiment_cfg = raw_config.get('experiment', {})
    dataset_cfg = raw_config.get('dataset', {})
    training_cfg = raw_config.get('training', {})
    logging_cfg = raw_config.get('logging', {})

    config = {}
    config['seed'] = raw_config.get('seed', experiment_cfg.get('seed', 42))
    config['device'] = raw_config.get(
        'device', experiment_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )

    config['data_root'] = raw_config.get('data_root', dataset_cfg.get('data_root', './data'))
    config['num_tasks'] = raw_config.get('num_tasks', dataset_cfg.get('num_tasks', 5))
    config['classes_per_task'] = raw_config.get(
        'classes_per_task', dataset_cfg.get('classes_per_task', 2)
    )

    config['num_epochs'] = raw_config.get('num_epochs', training_cfg.get('num_epochs', 200))
    config['batch_size'] = raw_config.get('batch_size', training_cfg.get('batch_size', 128))
    config['learning_rate'] = raw_config.get(
        'learning_rate', training_cfg.get('learning_rate', 0.1)
    )
    config['momentum'] = raw_config.get('momentum', training_cfg.get('momentum', 0.9))
    config['weight_decay'] = raw_config.get(
        'weight_decay', training_cfg.get('weight_decay', 5e-4)
    )
    config['scheduler'] = raw_config.get('scheduler', training_cfg.get('scheduler', 'cosine'))

    config['aggregators'] = raw_config.get('aggregators', [])

    config['output_dir'] = raw_config.get(
        'output_dir', logging_cfg.get('log_dir', './results')
    )
    config['save_checkpoints'] = raw_config.get(
        'save_checkpoints', logging_cfg.get('save_checkpoints', True)
    )
    config['checkpoint_dir'] = raw_config.get(
        'checkpoint_dir',
        os.path.join(config['output_dir'], 'checkpoints')
    )
    config['checkpoint_frequency'] = raw_config.get(
        'checkpoint_frequency', logging_cfg.get('checkpoint_frequency', 50)
    )
    config['eval_frequency'] = raw_config.get('eval_frequency', raw_config.get('evaluation', {}).get('eval_frequency', 10))

    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiTaskCIFAR10(Dataset):
    """CIFAR-10 dataset for multi-task learning with 5 binary tasks."""
    
    def __init__(self, root: str, train: bool = True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
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
    """ResNet-18 with multiple task-specific heads."""
    
    def __init__(self, num_tasks: int = 5, num_classes_per_task: int = 2):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        
        # Modify for CIFAR-10
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


class JacobianDescentTrainer:
    """Trainer for Jacobian Descent."""
    
    def __init__(self, model, aggregator, device, lr=0.1, momentum=0.9, weight_decay=5e-4):
        self.model = model
        self.aggregator = aggregator
        self.device = device
        self.num_tasks = len(getattr(model, 'heads', [])) or 5
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, batch_data, batch_labels):
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch_data)
        
        # Compute per-task losses
        losses = []
        for task_id, output in enumerate(outputs):
            loss = self.criterion(output, batch_labels[:, task_id])
            losses.append(loss)
        
        # Compute per-task gradients
        task_gradients = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            grads = []
            for param in self.model.parameters():
                if param.grad is None:
                    grads.append(torch.zeros_like(param).flatten())
                else:
                    grads.append(param.grad.flatten())
            task_gradients.append(torch.cat(grads))
        
        # Aggregate gradients
        aggregated_gradient = self.aggregator(task_gradients)
        
        # Apply aggregated gradient
        self.optimizer.zero_grad()
        idx = 0
        for param in self.model.parameters():
            num_params = param.numel()
            param.grad = aggregated_gradient[idx:idx+num_params].view_as(param)
            idx += num_params
        
        self.optimizer.step()
        
        return [loss.item() for loss in losses]
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = [0.0] * self.num_tasks
        num_batches = 0
        
        for batch_data, batch_labels, _ in tqdm(train_loader, desc="Training", leave=False):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            losses = self.train_step(batch_data, batch_labels)
            
            for i, loss in enumerate(losses):
                total_loss[i] += loss
            num_batches += 1
        
        return [loss / num_batches for loss in total_loss]
    
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        correct = [0] * self.num_tasks
        total = [0] * self.num_tasks
        
        with torch.no_grad():
            for batch_data, batch_labels, _ in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                
                for task_id, output in enumerate(outputs):
                    _, predicted = output.max(1)
                    correct[task_id] += (predicted == batch_labels[:, task_id]).sum().item()
                    total[task_id] += batch_labels.size(0)
        
        accuracies = [100.0 * c / t for c, t in zip(correct, total)]
        return accuracies


def run_experiment(config):
    """Run full experiment."""
    set_seed(config.get('seed', 42))
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Datasets
    data_root = config.get('data_root', './data')
    os.makedirs(data_root, exist_ok=True)
    
    train_dataset = MultiTaskCIFAR10(data_root, train=True, transform=transform_train)
    test_dataset = MultiTaskCIFAR10(data_root, train=False, transform=transform_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 128), 
        shuffle=True, 
        num_workers=2,
        pin_memory=(device == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.get('batch_size', 128), 
        shuffle=False, 
        num_workers=2,
        pin_memory=(device == 'cuda')
    )
    
    # Run experiments for each aggregator
    results = {}
    num_epochs = config.get('num_epochs', 200)
    
    for agg_config in config.get('aggregators', []):
        agg_name = agg_config['name']
        agg_type = agg_config['type']
        
        print(f"\n{'='*60}")
        print(f"Training with {agg_name} aggregator")
        print(f"{'='*60}")
        
        # Create model
        model = ResNet18MultiTask(
            num_tasks=config.get('num_tasks', 5),
            num_classes_per_task=config.get('classes_per_task', 2)
        ).to(device)
        
        # Create aggregator
        aggregator = get_aggregator(agg_type, **{k: v for k, v in agg_config.items() if k not in ['name', 'type']})
        
        # Create trainer
        trainer = JacobianDescentTrainer(
            model, aggregator, device,
            lr=config.get('learning_rate', 0.1),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4)
        )
        
        # Learning rate scheduler
        scheduler_name = str(config.get('scheduler', 'cosine')).lower()
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=num_epochs
            )
        else:
            scheduler = None
        
        # Training history
        history = {
            'train_loss': [],
            'test_acc': [],
            'per_task_acc': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            train_losses = trainer.train_epoch(train_loader)
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            test_accs = trainer.evaluate(test_loader)
            avg_test_acc = sum(test_accs) / len(test_accs)
            
            history['train_loss'].append(avg_train_loss)
            history['test_acc'].append(avg_test_acc)
            history['per_task_acc'].append(test_accs)
            
            if (epoch + 1) % max(1, int(config.get('eval_frequency', 10))) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={avg_train_loss:.4f}, "
                      f"Avg Acc={avg_test_acc:.2f}%, "
                      f"Tasks={[f'{acc:.1f}' for acc in test_accs]}")
            
            if scheduler is not None:
                scheduler.step()
        
        results[agg_name] = history
        
        # Save model checkpoint
        if config.get('save_checkpoints', True):
            checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'history': history
            }, os.path.join(checkpoint_dir, f'{agg_name}_final.pth'))
    
    return results


def plot_results(results, save_path='results.png'):
    """Plot and save results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    for agg_name, history in results.items():
        axes[0].plot(history['train_loss'], label=agg_name, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Test accuracy
    for agg_name, history in results.items():
        axes[1].plot(history['test_acc'], label=agg_name, linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1].set_title('Test Accuracy over Epochs', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nResults plot saved to {save_path}")
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for agg_name, history in results.items():
        final_acc = history['test_acc'][-1]
        final_task_accs = history['per_task_acc'][-1]
        print(f"\n{agg_name}:")
        print(f"  Average Accuracy: {final_acc:.2f}%")
        print(f"  Per-task Accuracies: {[f'{acc:.2f}%' for acc in final_task_accs]}")
        print(f"  Min Task Accuracy: {min(final_task_accs):.2f}%")
        print(f"  Max Task Accuracy: {max(final_task_accs):.2f}%")
        print(f"  Fairness (std): {np.std(final_task_accs):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Run Jacobian Descent experiments')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        raw_config = yaml.safe_load(f)
    config = normalize_config(raw_config or {})
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Run experiment
    results = run_experiment(config)
    
    # Save results
    output_dir = config.get('output_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(results, os.path.join(output_dir, 'results.png'))
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
