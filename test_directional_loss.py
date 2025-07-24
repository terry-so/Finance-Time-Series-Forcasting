#!/usr/bin/env python3

"""
Test script for directional loss implementation
"""

from utils.tools import get_loss_function
from utils.directional_loss import DirectionalLoss, WeightedDirectionalLoss
import sys
import os
import torch
import numpy as np

# Add the project root to Python path
sys.path.append('/workspaces/Finance-Time-Series-Forcasting')


def test_directional_loss():
    """Test the directional loss functions"""
    print("Testing Directional Loss Implementation...")

    # Create sample data
    batch_size, pred_len, features = 8, 5, 1

    # Ground truth: increasing trend
    true = torch.randn(batch_size, pred_len, features)
    true[:, 1:, :] = true[:, :-1, :] + 0.1  # Increasing trend

    # Prediction 1: correct direction
    pred_correct = true + torch.randn_like(true) * 0.05

    # Prediction 2: wrong direction
    pred_wrong = true * -1 + torch.randn_like(true) * 0.05

    # Last values for directional calculation
    last_values = torch.randn(batch_size, features)

    print(
        f"Data shapes: true={true.shape}, pred={pred_correct.shape}, last={last_values.shape}")

    # Test DirectionalLoss
    print("\n1. Testing DirectionalLoss...")
    dir_loss = DirectionalLoss(alpha=0.5, beta=1.0, base_loss='mse')

    loss_correct = dir_loss(pred_correct, true, last_values)
    loss_wrong = dir_loss(pred_wrong, true, last_values)

    print(f"Loss with correct direction: {loss_correct.item():.4f}")
    print(f"Loss with wrong direction: {loss_wrong.item():.4f}")
    print(f"Wrong direction loss is higher: {loss_wrong > loss_correct}")

    # Test WeightedDirectionalLoss
    print("\n2. Testing WeightedDirectionalLoss...")
    weighted_loss = WeightedDirectionalLoss(
        alpha=0.5, beta=1.0, gamma=0.1, base_loss='mse')

    w_loss_correct = weighted_loss(pred_correct, true, last_values)
    w_loss_wrong = weighted_loss(pred_wrong, true, last_values)

    print(f"Weighted loss with correct direction: {w_loss_correct.item():.4f}")
    print(f"Weighted loss with wrong direction: {w_loss_wrong.item():.4f}")
    print(f"Wrong direction loss is higher: {w_loss_wrong > w_loss_correct}")

    # Test get_loss_function
    print("\n3. Testing get_loss_function...")

    losses_to_test = [
        ('mae', {}),
        ('mse', {}),
        ('directional_mae', {'alpha': 0.6, 'beta': 1.2}),
        ('directional_mse', {'alpha': 0.4, 'beta': 0.8}),
        ('weighted_directional', {'alpha': 0.5, 'beta': 1.0, 'gamma': 0.2})
    ]

    for loss_name, kwargs in losses_to_test:
        criterion = get_loss_function(loss_name, **kwargs)
        print(f"✓ {loss_name}: {type(criterion).__name__}")

        # Test the loss function
        if 'directional' in loss_name:
            loss_val = criterion(pred_correct, true, last_values)
        else:
            loss_val = criterion(pred_correct, true)
        print(f"  Sample loss: {loss_val.item():.4f}")

    print("\n4. Testing period-to-period directional loss...")
    # Test without last_values (period-to-period)
    dir_loss_period = DirectionalLoss(alpha=0.5, beta=1.0, base_loss='mae')
    loss_period = dir_loss_period(pred_correct, true)
    print(f"Period-to-period directional loss: {loss_period.item():.4f}")

    print("\n✅ All directional loss tests completed successfully!")


def test_integration_example():
    """Show example of how to use directional loss in training"""
    print("\n" + "="*50)
    print("INTEGRATION EXAMPLE")
    print("="*50)

    print("""
# Example: Using directional loss in training command:
python run.py --is_training 1 --model_id "directional_test" --model xPatch \\
    --data custom --data_path aapl_OHLCV.csv --features MS --target Close \\
    --loss directional_mae --directional_alpha 0.5 --directional_beta 1.0

# Available loss functions:
- 'mae': Standard Mean Absolute Error
- 'mse': Standard Mean Squared Error  
- 'directional_mae': MAE + directional penalty
- 'directional_mse': MSE + directional penalty
- 'weighted_directional': Magnitude-weighted directional loss

# Directional loss parameters:
- directional_alpha: Weight for base loss (default: 0.5)
- directional_beta: Weight for directional penalty (default: 1.0) 
- directional_gamma: Weight for magnitude penalty in weighted version (default: 0.1)
""")


if __name__ == "__main__":
    test_directional_loss()
    test_integration_example()
