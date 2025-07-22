from models.xPatch import Model
import torch
import sys
sys.path.append('/workspaces/Finance-Time-Series-Forcasting')


class TestArgs:
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


args = TestArgs()
model = Model(args)

print("Testing shapes...")
test_input = torch.randn(2, args.seq_len, args.enc_in)
print(f"Original input shape: {test_input.shape}")

# Get the network for debugging
net = model.net

# Simulate the preprocessing
s = test_input
t = test_input

print(f"Before permute - s: {s.shape}")
s = s.permute(0, 2, 1)
print(f"After permute - s: {s.shape}")

B = s.shape[0]
C = s.shape[1]
I = s.shape[2]
s = torch.reshape(s, (B*C, I))
print(f"After reshape - s: {s.shape}")

# Patching
if net.padding_patch == 'end':
    s = net.padding_patch_layer(s)
    print(f"After padding - s: {s.shape}")

s = s.unfold(dimension=-1, size=net.patch_len, step=net.stride)
print(f"After unfold (patching) - s: {s.shape}")

# Patch embedding
s = net.fc1(s)
print(f"After patch embedding - s: {s.shape}")

print(f"LSTM expects 3D input, got {len(s.shape)}D with shape {s.shape}")
print(f"LSTM input_size: {net.patch_lstm.input_size}")
print(f"Expected LSTM input format: [batch_size, sequence_length, input_size]")
print(
    f"Current format: [batch*channel={B*C}, patch_num, embedding_dim={net.dim}]")
