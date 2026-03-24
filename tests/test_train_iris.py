"""
Iris Classification using MLP with Second-Order Optimizer

This script demonstrates training a simple MLP on the classic iris dataset
using the CUDA-accelerated implementation with second-order optimization.

Dataset: Iris (4 features, 3 classes, 150 samples)
Model: MLP with 8 hidden units
Optimizer: Damped Newton's method (direct solve)

Author: Dy164
Date: 2025-12-05
"""

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path

# Import our custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsplat2ct import (
    MLPClassificationCUDA,
    MLPParams,
    MLPOptimizerConfig,
    MLPSecondOrderOptimizer,
    Data,
)


def load_iris_data():
    """Load and preprocess iris dataset."""
    # Load data
    iris = load_iris()
    X = iris.data  # (150, 4)
    y = iris.target  # (150,) - class indices 0, 1, 2
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Convert labels to one-hot encoding
    y_train_onehot = torch.zeros(y_train.shape[0], 3)
    y_train_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)
    
    y_test_onehot = torch.zeros(y_test.shape[0], 3)
    y_test_onehot.scatter_(1, y_test.unsqueeze(1), 1.0)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {y_train_onehot.shape[1]}")
    
    return X_train, y_train_onehot, X_test, y_test_onehot, y_train, y_test


def initialize_params(i=4, m=8, o=3, device='cuda'):
    """Initialize MLP parameters."""
    # Xavier initialization
    w1 = torch.randn(m, i, device=device) * np.sqrt(2.0 / i)
    b1 = torch.zeros(m, 1, device=device)
    w2 = torch.randn(o, m, device=device) * np.sqrt(2.0 / m)
    b2 = torch.zeros(o, 1, device=device)
    
    # Create MLPParams
    params = MLPParams(w1=w1, b1=b1, w2=w2, b2=b2)
    
    return params


def compute_accuracy(predictions, targets_indices):
    """Compute classification accuracy.
    
    Args:
        predictions: (batch_size, num_classes) probabilities
        targets_indices: (batch_size,) class indices
    
    Returns:
        accuracy: float between 0 and 1
    """
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == targets_indices).sum().item()
    total = targets_indices.shape[0]
    return correct / total


def train_epoch(model, optimizer, params, X_train, y_train, batch_size=32):
    """Train for one epoch with mini-batches."""
    num_samples = X_train.shape[0]
    indices = torch.randperm(num_samples)
    
    total_loss = 0.0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices].cuda()
        y_batch = y_train[batch_indices].cuda()
        
        # Create data container
        data = Data()
        data.set('inputs', X_batch)
        data.set('targets', y_batch)
        
        # Forward + backward to get gradients and Hessians
        fwd_result, grad_result, hess_result = model.forward_backward2(
            params, data, config=None
        )
        
        # Add gradients and Hessians to optimizer
        optimizer.add_grad_hess(grad_result, hess_result)
        
        # Update parameters
        optimizer.step()
        
        # Zero gradients for next iteration
        optimizer.zero_grad()
        
        total_loss += fwd_result.outputs['loss'].item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, params, X_test, y_test_onehot, y_test_indices):
    """Evaluate model on test set."""
    X_test_cuda = X_test.cuda()
    
    # Create data container without targets (testing mode)
    data = Data()
    data.set('inputs', X_test_cuda)
    # Note: Don't set 'targets' - this will trigger inference mode
    
    # Forward pass only
    fwd_result = model.forward_only(params, data, config=None)
    
    predictions = fwd_result.outputs['predictions'].cpu()
    accuracy = compute_accuracy(predictions, y_test_indices)
    
    # Also compute loss for monitoring
    data.set('targets', y_test_onehot.cuda())
    fwd_result_with_loss = model.forward_only(params, data, config=None)
    test_loss = fwd_result_with_loss.outputs['loss'].item()
    
    return accuracy, test_loss


def main():
    print("=" * 70)
    print("Iris Classification with Second-Order Optimization")
    print("=" * 70)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "=" * 70)
    print("Loading and preprocessing data...")
    print("=" * 70)
    X_train, y_train, X_test, y_test, y_train_idx, y_test_idx = load_iris_data()
    
    # Initialize model
    print("\n" + "=" * 70)
    print("Initializing model...")
    print("=" * 70)
    i, m, o = 4, 8, 3
    print(f"Architecture: {i} → {m} → {o}")
    
    params = initialize_params(i=i, m=m, o=o, device='cuda')
    model = MLPClassificationCUDA(i=i, m=m, o=o)
    
    # Initialize optimizer
    print("\nInitializing optimizer...")
    optimizer_config = MLPOptimizerConfig(
        lr=0.01,
        damping=1e-3
    )
    optimizer = MLPSecondOrderOptimizer(params, optimizer_config)
    print(f"Learning rates: {optimizer_config.lr_map}")
    print(f"Damping factors: {optimizer_config.damping_map}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    num_epochs = 800
    batch_size = 2
    
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train one epoch
        train_loss = train_epoch(model, optimizer, params, X_train, y_train, batch_size)
        
        # Evaluate
        test_acc, test_loss = evaluate(model, params, X_test, y_test, y_test_idx)
        
        epoch_time = time.time() - start_time
        
        # Update best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc*100:.2f}% | "
                  f"Time: {epoch_time:.3f}s")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    final_acc, final_loss = evaluate(model, params, X_test, y_test, y_test_idx)
    print(f"Test Accuracy: {final_acc*100:.2f}%")
    print(f"Test Loss: {final_loss:.4f}")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    
    # Per-class analysis
    print("\n" + "=" * 70)
    print("Per-Class Analysis")
    print("=" * 70)
    
    X_test_cuda = X_test.cuda()
    data = Data()
    data.set('inputs', X_test_cuda)
    fwd_result = model.forward_only(params, data, config=None)
    predictions = fwd_result.outputs['predictions'].cpu()
    pred_classes = torch.argmax(predictions, dim=1)
    
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    for i, class_name in enumerate(class_names):
        mask = y_test_idx == i
        if mask.sum() > 0:
            class_acc = (pred_classes[mask] == i).float().mean().item()
            print(f"{class_name:12s}: {class_acc*100:.2f}% ({mask.sum()} samples)")


if __name__ == '__main__':
    main()
