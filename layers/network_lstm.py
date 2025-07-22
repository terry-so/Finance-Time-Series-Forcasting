import torch
from torch import nn


class NetworkLSTM(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 lstm_hidden_size=None, lstm_layers=1, lstm_dropout=0.1, lstm_bidirectional=False):
        super(NetworkLSTM, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)

        # LSTM Configuration
        self.lstm_hidden_size = lstm_hidden_size or self.dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_bidirectional = lstm_bidirectional

        # Calculate effective hidden size (bidirectional doubles output)
        effective_hidden = self.lstm_hidden_size * \
            (2 if lstm_bidirectional else 1)

        # LSTM for processing patch sequences
        self.patch_lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )

        # Projection layer to match dimensions
        if effective_hidden != self.dim:
            self.lstm_proj = nn.Linear(effective_hidden, self.dim)
        else:
            self.lstm_proj = nn.Identity()

        # Layer normalization for better training stability
        self.lstm_norm = nn.LayerNorm(self.dim)

        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len * 2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len)

        self.fc7 = nn.Linear(pred_len, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B*C, I))  # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I))  # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]

        # Patch Embedding
        s = self.fc1(s)

        # LSTM processing after patch embedding (KEY ENHANCEMENT)
        # Store original for residual connection
        s_residual = s

        # LSTM expects [Batch, Sequence, Features] format
        # s is currently [Batch*Channel, Patch_num, dim]
        lstm_out, _ = self.patch_lstm(s)

        # Project to match dimensions if needed
        lstm_out = self.lstm_proj(lstm_out)

        # Residual connection + Layer normalization
        s = self.lstm_norm(s_residual + lstm_out)

        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatination
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]

        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x
