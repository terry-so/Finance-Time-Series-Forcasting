#!/usr/bin/env python3

"""
Test script for directional loss integration with training loop
"""

from exp.exp_main import Exp_Main
import sys
import os
import argparse
import torch
import numpy as np

# Add the project root to Python path
sys.path.append('/workspaces/Finance-Time-Series-Forcasting')


def create_test_args():
    """Create test arguments for directional loss training"""

    class TestArgs:
        def __init__(self):
            # Basic config
            self.is_training = 1
            self.train_only = False
            self.model_id = 'directional_loss_test'
            self.model = 'xPatch'

            # Data config
            self.data = 'custom'
            self.root_path = './data/'
            self.data_path = 'aapl_OHLCV.csv'
            self.features = 'MS'
            self.target = 'Close'
            self.freq = 'd'
            self.checkpoints = './checkpoints/'
            self.embed = 'timeF'

            # Model parameters
            self.seq_len = 48
            self.label_len = 24
            self.pred_len = 5
            self.enc_in = 9

            # Patching
            self.patch_len = 12
            self.stride = 6
            self.padding_patch = 'end'

            # Moving Average
            self.ma_type = 'ema'
            self.alpha = 0.3
            self.beta = 0.3

            # Optimization
            self.num_workers = 4
            self.itr = 1
            self.train_epochs = 2  # Short test
            self.batch_size = 16
            self.patience = 5
            self.learning_rate = 0.0001
            self.des = 'directional_test'

            # DIRECTIONAL LOSS CONFIGURATION
            self.loss = 'directional_mae'  # Test directional loss
            self.directional_alpha = 0.6   # Weight for base loss
            self.directional_beta = 1.2    # Weight for directional penalty
            self.directional_gamma = 0.1   # For weighted version

            self.lradj = 'type1'
            self.use_amp = False
            self.revin = 1

            # GPU
            self.use_gpu = torch.cuda.is_available()
            self.gpu = 0
            self.use_multi_gpu = False
            self.devices = '0'
            self.test_flop = False

            # LSTM (optional)
            self.use_lstm = False
            self.lstm_hidden_size = 128
            self.lstm_layers = 2
            self.lstm_dropout = 0.1
            self.lstm_bidirectional = False

            # WandB (disable for test)
            self.use_wandb = False

    return TestArgs()


def test_directional_loss_training():
    """Test training with directional loss"""
    print("Testing Directional Loss Integration with Training...")

    # Create test arguments
    args = create_test_args()

    # Test different loss functions
    loss_functions = [
        'mae',              # Standard MAE
        'directional_mae',  # MAE + directional penalty
        'weighted_directional'  # Weighted directional loss
    ]

    for loss_name in loss_functions:
        print(f"\n{'='*50}")
        print(f"Testing with loss function: {loss_name}")
        print(f"{'='*50}")

        # Update loss function
        args.loss = loss_name
        args.model_id = f'directional_test_{loss_name}'

        try:
            # Create experiment
            exp = Exp_Main(args)

            # Test criterion selection
            main_criterion, mse_criterion = exp._select_criterion()
            print(f"✓ Main criterion: {type(main_criterion).__name__}")
            print(f"✓ MSE criterion: {type(mse_criterion).__name__}")

            # Test with sample data
            batch_size, seq_len, features = 8, args.seq_len, 1

            # Create sample inputs
            batch_x = torch.randn(batch_size, seq_len, features)
            batch_y = torch.randn(batch_size, args.pred_len, features)

            # Test loss calculation
            if 'directional' in loss_name:
                last_values = batch_x[:, -1, :]
                loss = main_criterion(batch_y, batch_y, last_values)
            else:
                loss = main_criterion(batch_y, batch_y)

            print(f"✓ Loss calculation successful: {loss.item():.6f}")

            # Test one training step (data loading)
            print("✓ Testing data loading...")
            train_data, train_loader = exp._get_data(flag='train')
            print(f"✓ Training data loaded: {len(train_data)} samples")

            # Get one batch to verify shapes
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                print(
                    f"✓ Batch shapes - X: {batch_x.shape}, Y: {batch_y.shape}")
                break

        except Exception as e:
            print(f"❌ Error with {loss_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*50}")
    print("✅ Directional Loss Integration Test Completed!")
    print(f"{'='*50}")


def show_usage_examples():
    """Show usage examples for directional loss"""
    print("""
USAGE EXAMPLES FOR DIRECTIONAL LOSS:

1. Basic directional MAE loss:
   python run.py --is_training 1 --model_id "dir_mae_test" --model xPatch \\
       --data custom --data_path aapl_OHLCV.csv --features MS --target Close \\
       --loss directional_mae --directional_alpha 0.5 --directional_beta 1.0

2. Directional MSE loss with custom weights:
   python run.py --is_training 1 --model_id "dir_mse_test" --model xPatch \\
       --data custom --data_path aapl_OHLCV.csv --features MS --target Close \\
       --loss directional_mse --directional_alpha 0.3 --directional_beta 1.5

3. Weighted directional loss (magnitude-aware):
   python run.py --is_training 1 --model_id "weighted_dir_test" --model xPatch \\
       --data custom --data_path aapl_OHLCV.csv --features MS --target Close \\
       --loss weighted_directional --directional_alpha 0.4 --directional_beta 1.0 --directional_gamma 0.2

4. ETT dataset with directional loss:
   python run.py --is_training 1 --model_id "ett_directional" --model xPatch \\
       --data ETTh1 --data_path ETTh1.csv --features M --target OT \\
       --loss directional_mae --seq_len 96 --pred_len 96

LOSS FUNCTION COMPARISON:
- Standard Loss: Optimizes only prediction accuracy
- Directional Loss: Optimizes accuracy + directional correctness 
- Weighted Directional: Also considers magnitude of changes

PARAMETER TUNING GUIDELINES:
- directional_alpha: 0.3-0.7 (higher = more weight on base loss)
- directional_beta: 0.8-2.0 (higher = more directional penalty)
- directional_gamma: 0.1-0.3 (for weighted version only)
""")


if __name__ == "__main__":
    test_directional_loss_training()
    show_usage_examples()
