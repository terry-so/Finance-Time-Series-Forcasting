import torch
import torch.nn as nn
import math

from layers.decoders import LSTMDecoder
from layers.decomp import DECOMP
from layers.network_lstm import NetworkLSTM
# from layers.network_mlp import NetworkMLP # For ablation study with MLP-only stream
# from layers.network_cnn import NetworkCNN # For ablation study with CNN-only stream
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        self.pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = NetworkLSTM(seq_len, self.pred_len, patch_len, stride, padding_patch)
        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream
        context_vector_size = configs.pred_len * 2
        self.decoder = LSTMDecoder(
            input_size=context_vector_size,
            hidden_size=context_vector_size,
            output_size=configs.enc_in,
            pred_len=configs.pred_len
        )

    def forward(self, x):
        B, L, C = x.shape 
        
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        seasonal_init, trend_init = self.decomp(x)
        context_vector = self.net(seasonal_init, trend_init)
        x = self.decoder(context_vector) # Output shape is now [B*C, pred_len, enc_in]
        

        x = x.view(B, C, self.pred_len, -1).squeeze()
        x = x.permute(0, 2, 1, 3)
        x = torch.diagonal(x, offset=0, dim1=-2, dim2=-1)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x