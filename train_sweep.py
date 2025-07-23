# In file: train_sweep.py

import torch
import wandb
import os

# Import your classes
from exp.exp_main import Exp_Main
from utils.tools_lstm import Args # Assuming you've moved your Args class to utils/tools.py

# --- Main Training Function ---
def main():
    # 1. Initialize W&B. The agent will pass the hyperparameters.
    wandb.init()

    # 2. Copy the W&B config into an Args object
    # This allows us to use W&B's parameters with your existing code structure
    args = Args() # Your base Args class
    
    # Update args with the values from the W&B sweep for this run
    args.learning_rate = wandb.config.learning_rate
    args.batch_size = wandb.config.batch_size
    args.seq_len = wandb.config.seq_len
    args.dropout = wandb.config.dropout

    if args.model == 'xPatch':
        args.experiment_notes = 'Training xPatch'
        # If xPatch has specific *architecture* parameters that differ from LSTM's, set them here
        # Example: args.d_model = 128 # If this was different for LSTM
    elif args.model == 'xPatch_LSTM':
        args.experiment_notes = 'Training xPatch_LSTM'

    
    
    # Update any dependent arguments
    args.size = [args.seq_len, args.label_len, args.pred_len]
    
    # Give the run a more descriptive name
    args.model_id = f'Sweep_LR_AAPL_LSTM_{args.learning_rate:.6f}_BS_{args.batch_size}_SL_{args.seq_len}_PL_{args.pred_len}_DO_{args.dropout}'

    # 3. Run your experiment as before
    print(f"--- Starting Run with: LR={args.learning_rate}, BS={args.batch_size}, SeqLen={args.seq_len} ---")
    exp = Exp_Main(args)
    exp.train(args.model_id)

if __name__ == '__main__':
    main()