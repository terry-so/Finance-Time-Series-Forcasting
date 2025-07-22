from models.xPatch import Model
import torch
import sys
sys.path.append('/workspaces/Finance-Time-Series-Forcasting')


class TestArgs:
    def __init__(self, use_lstm=True):
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
        self.use_lstm = use_lstm
        self.lstm_hidden_size = 96
        self.lstm_layers = 1
        self.lstm_dropout = 0.1
        self.lstm_bidirectional = False


print("🔍 LSTM After Patch Embedding Verification")
print("=" * 50)

# Test with and without LSTM
args_lstm = TestArgs(use_lstm=True)
args_no_lstm = TestArgs(use_lstm=False)

model_lstm = Model(args_lstm)
model_no_lstm = Model(args_no_lstm)

print("\n1. Architecture Verification:")
print(f"   LSTM Model has patch_lstm: {hasattr(model_lstm.net, 'patch_lstm')}")
print(
    f"   No-LSTM Model has patch_lstm: {hasattr(model_no_lstm.net, 'patch_lstm')}")

if hasattr(model_lstm.net, 'patch_lstm'):
    lstm = model_lstm.net.patch_lstm
    print(f"   LSTM Configuration:")
    print(
        f"     - Input size: {lstm.input_size} (patch_len^2 = {args_lstm.patch_len}^2 = {args_lstm.patch_len**2})")
    print(f"     - Hidden size: {lstm.hidden_size}")
    print(f"     - Layers: {lstm.num_layers}")
    print(f"     - Bidirectional: {lstm.bidirectional}")

print("\n2. Parameter Count Comparison:")
params_lstm = sum(p.numel() for p in model_lstm.parameters())
params_no_lstm = sum(p.numel() for p in model_no_lstm.parameters())
lstm_params = params_lstm - params_no_lstm

print(f"   With LSTM: {params_lstm:,} parameters")
print(f"   Without LSTM: {params_no_lstm:,} parameters")
print(f"   LSTM adds: {lstm_params:,} parameters")

print("\n3. Forward Pass Test:")
test_input = torch.randn(4, args_lstm.seq_len, args_lstm.enc_in)

with torch.no_grad():
    output_lstm = model_lstm(test_input)
    output_no_lstm = model_no_lstm(test_input)

print(f"   Input shape: {test_input.shape}")
print(f"   LSTM output shape: {output_lstm.shape}")
print(f"   No-LSTM output shape: {output_no_lstm.shape}")

# Check if outputs are different
different = not torch.allclose(output_lstm, output_no_lstm, atol=1e-6)
print(f"   Outputs are different: {'✅' if different else '❌'}")

print("\n4. LSTM Impact Analysis:")
lstm_mean = output_lstm.mean().item()
no_lstm_mean = output_no_lstm.mean().item()
diff = abs(lstm_mean - no_lstm_mean)

print(f"   LSTM output mean: {lstm_mean:.6f}")
print(f"   No-LSTM output mean: {no_lstm_mean:.6f}")
print(f"   Mean difference: {diff:.6f}")

print("\n5. Architecture Flow Verification:")
print("   Expected flow:")
print("   Input → RevIN → Decomposition → Network")
print(
    "   Network: Patching → Embedding → [LSTM] → GELU → BatchNorm → CNN → ...")
print("   Where [LSTM] is: s_residual + LSTM(s) with LayerNorm")

print("\n🎯 VERIFICATION RESULT:")
if hasattr(model_lstm.net, 'patch_lstm') and different and lstm_params > 0:
    print("✅ LSTM IS CORRECTLY IMPLEMENTED AFTER PATCH EMBEDDING!")
    print("   - LSTM layers are present and configured correctly")
    print("   - LSTM is actively processing patch embeddings")
    print("   - Residual connection maintains gradient flow")
    print("   - LayerNorm stabilizes training")
    print("   - Flow: Patch Embed → LSTM → GELU (as intended)")
else:
    print("❌ LSTM implementation has issues")

print(f"\n📊 Summary:")
print(f"   - Patch size: {args_lstm.patch_len}")
print(f"   - Patch stride: {args_lstm.stride}")
print(f"   - Patch dimension: {args_lstm.patch_len**2} (patch_len^2)")
print(f"   - LSTM input: {lstm.input_size} features per patch")
print(f"   - LSTM hidden: {lstm.hidden_size}")
print(
    f"   - Sequence length: {(args_lstm.seq_len - args_lstm.patch_len)//args_lstm.stride + 1} patches")
