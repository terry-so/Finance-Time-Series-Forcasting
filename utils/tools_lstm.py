# In file: utils/tools.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math # Added for a function in Exp_Main

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    # ... (all other lr adjust types from the original file) ...
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# ... (dotdict, StandardScaler, and visual functions from the original file) ...
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

# ==============================================================================
# ADD YOUR ARGS CLASS HERE
# ==============================================================================
class Args:
    def __init__(self):
        self.is_training = 1
        self.train_only = False
        # self.model_id = 'AAPL_LSTM_v1' # This will be set by the sweep script
        self.model = 'xPatch' #xPatch/xPatch_LSTM

        self.data = 'custom'
        self.scale = True
        self.root_path = './data/'
        self.data_path = 'aapl_OHLCV.csv'
        self.features = 'MS'
        self.target = 'Close'
        self.freq = 'B'
        self.checkpoints = './checkpoints/'
        self.embed = 'timeF'
        self.timeenc = 0

        # --- Hyperparameters ---
        # NOTE: The sweep will override some of these (e.g., learning_rate, batch_size)
        self.label_len = 48
        self.pred_len = 5
        self.enc_in = 9
        self.patch_len = 12
        self.stride = 6
        self.padding_patch = 'end'
        self.ma_type = 'ema'
        self.alpha = 0.2
        self.beta = 0.2
        self.num_workers = 8
        self.itr = 1
        self.train_epochs = 100
        self.patience = 15
        self.loss = 'mae'
        self.lradj = 'type1'
        self.revin = 1
        
        # --- GPU ---
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'

        # --- Wandb configuration ---
        self.use_wandb = True  # Enable/disable wandb logging
        self.wandb_entity = 'xplstm' # <-- IMPORTANT: Change to your username or team name
        self.wandb_project = 'CS7643-GroupProject'
        self.experiment_notes = 'vanilla xpatch'
        
        # --- Model Specific ---
        self.d_model = 128
        self.d_ff = 256
        self.e_layers = 3
        self.k = 3
        self.decomp = 0