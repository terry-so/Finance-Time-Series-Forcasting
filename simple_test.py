from models.xPatch import Model
import torch
import sys
import os

print("Starting LSTM verification...")

# Add project root to path
sys.path.append('/workspaces/Finance-Time-Series-Forcasting')

print("Importing Model...")


class SimpleTestArgs:
    def __init__(self):
        self.seq_len = 48
        self.pred_len = 5
        self.enc_in = 7
        self.patch_len = 12
        self.stride = 6
        self.padding_patch = 'end'
        self.ma_type = 'ema'
        self.alpha = 0.2
        self.beta = 0.2
        self.revin = 1
        self.use_lstm = True
        self.lstm_hidden_size = 96
        self.lstm_layers = 1
        self.lstm_dropout = 0.1
        self.lstm_bidirectional = False


print("Creating test args...")
args = SimpleTestArgs()

print("Creating model...")
model = Model(args)

print("Checking LSTM components...")
print(f"Model has patch_lstm: {hasattr(model.net, 'patch_lstm')}")
print(f"Model has lstm_proj: {hasattr(model.net, 'lstm_proj')}")
print(f"Model has lstm_norm: {hasattr(model.net, 'lstm_norm')}")

if hasattr(model.net, 'patch_lstm'):
    lstm = model.net.patch_lstm
    print(f"LSTM input size: {lstm.input_size}")
    print(f"LSTM hidden size: {lstm.hidden_size}")
    print(f"LSTM layers: {lstm.num_layers}")

print("Creating test input...")
test_input = torch.randn(2, args.seq_len, args.enc_in)

print("Testing forward pass...")
with torch.no_grad():
    output = model(test_input)
    print(f"Output shape: {output.shape}")

print("✅ LSTM verification complete!")
