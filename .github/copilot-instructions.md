# Finance Time Series Forecasting with xPatch - Copilot Instructions

## Architecture Overview

This is a research implementation of **xPatch**, a hybrid architecture combining patching, CNN depthwise convolutions, and optional LSTM enhancement for financial time series forecasting. The core philosophy: decompose input sequences into patches, process them through parallel streams, and predict future values.

### Key Components
- **Models**: `models/xPatch.py` - Main model with optional LSTM via `layers/network_lstm.py`
- **Experiments**: `exp/exp_main.py` - Training/testing orchestration with extensive WandB integration
- **Data**: `data_provider/` - Handles ETT datasets and custom financial data (AAPL)
- **Core Logic**: `layers/network.py` - Patch embedding, CNN streams, and prediction heads

## Critical Workflows

### Training Commands
```bash
# Standard training
python run.py --is_training 1 --model_id "experiment_name" --model xPatch --data custom --data_path aapl_OHLCV.csv --features MS --target Close

# LSTM-enhanced training (set in notebook/sweep)
# Enable via: args.use_lstm = True, args.lstm_hidden_size = 128, args.lstm_layers = 2
```

### Hyperparameter Sweeps
Use `finetune.ipynb` for WandB sweeps. Key pattern:
1. Define sweep config with nested parameter distributions
2. Critical type conversions: `int(float(str(value)))` to handle WandB/PyTorch tensor conflicts
3. Parameter validation: ensure `pred_len < seq_len`, `patch_len >= 4`, `num_patches > 0`

## Project-Specific Patterns

### Data Handling
- **ETT datasets**: Electricity Transformer Temperature data (ETTh1.csv)
- **Custom data**: Apple stock (aapl_OHLCV.csv) with 9 features including time encodings
- **Feature modes**: `MS` (multivariate→univariate), `M` (multivariate→multivariate), `S` (univariate→univariate)

### Loss Architecture
```python
# Arctangent temporal weighting (not standard MSE)
ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(pred_len)])
weighted_outputs = outputs * ratio
weighted_targets = targets * ratio
loss = mae_criterion(weighted_outputs, weighted_targets)
```

### Patch Configuration
```python
# Critical calculation for valid configurations
num_patches = (seq_len - patch_len) // stride + 1
# Must ensure num_patches > 0 and reasonable overlap
```

### WandB Integration
Comprehensive logging in `exp_main.py`:
- Batch-level metrics every 20 iterations
- Epoch-level train/val/test losses
- Sample prediction visualizations
- Error distribution histograms
- Model architecture watching

## Common Pitfalls

1. **Type Conversions**: WandB sweep parameters arrive as tensors; convert via `int(float(str(value)))`
2. **LSTM Parameters**: Must be native Python types, not tensors, or model initialization fails
3. **Patch Validation**: Always validate `(seq_len - patch_len) // stride + 1 > 0`
4. **Checkpoints**: Stored in `./checkpoints/{experiment_name}/checkpoint.pth`
5. **Results**: Test outputs saved to `./test_results/{experiment_name}/`

## Key Files for Understanding
- `run.py` - Command-line interface and argument parsing
- `exp/exp_main.py` - Training loop with temporal weighting and WandB logging  
- `models/xPatch.py` - Model architecture with optional LSTM branching
- `layers/decomp.py` - Moving average decomposition (EMA/DEMA)
- `finetune.ipynb` - Sweep configuration and type handling examples

## Data Pipeline
1. Load via `data_provider/data_factory.py`
2. Apply StandardScaler and RevIN normalization
3. Generate time features if `embed='timeF'`
4. Split into train/val/test with sliding windows
5. Patch sequences and apply temporal weighting during training

When debugging training issues, check: parameter types (tensors vs native Python), patch configuration validity, and WandB initialization status.
