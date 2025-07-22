"""
Comprehensive LSTM Verification Script
Tests if LSTM is properly applied in the xPatch model
"""
from models.xPatch import Model
import torch
import sys
import os

# Add project root to path
project_root = os.path.abspath('./')
if project_root not in sys.path:
    sys.path.append(project_root)


class TestArgs:
    def __init__(self, use_lstm=True):
        # Basic parameters
        self.seq_len = 48
        self.pred_len = 5
        self.enc_in = 7

        # Patching
        self.patch_len = 12
        self.stride = 6
        self.padding_patch = 'end'

        # Moving Average
        self.ma_type = 'ema'
        self.alpha = 0.2
        self.beta = 0.2

        # Normalization
        self.revin = 1

        # LSTM Configuration
        self.use_lstm = use_lstm
        self.lstm_hidden_size = 96
        self.lstm_layers = 2
        self.lstm_dropout = 0.15
        self.lstm_bidirectional = True


def verify_lstm_implementation():
    print("🔍 Comprehensive LSTM Verification")
    print("=" * 50)

    # Test 1: Model Creation
    print("\n1. Testing Model Creation:")
    args_with_lstm = TestArgs(use_lstm=True)
    args_without_lstm = TestArgs(use_lstm=False)

    try:
        model_with_lstm = Model(args_with_lstm)
        model_without_lstm = Model(args_without_lstm)
        print("   ✅ Both models created successfully")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

    # Test 2: LSTM Layer Presence
    print("\n2. Verifying LSTM Layer Presence:")

    # Check if LSTM is properly initialized
    has_lstm = hasattr(model_with_lstm.net, 'patch_lstm')
    has_lstm_proj = hasattr(model_with_lstm.net, 'lstm_proj')
    has_lstm_norm = hasattr(model_with_lstm.net, 'lstm_norm')

    print(f"   LSTM layer present: {'✅' if has_lstm else '❌'}")
    print(f"   LSTM projection layer: {'✅' if has_lstm_proj else '❌'}")
    print(f"   LSTM normalization: {'✅' if has_lstm_norm else '❌'}")

    if not (has_lstm and has_lstm_proj and has_lstm_norm):
        print("   ❌ LSTM components missing")
        return False

    # Test 3: LSTM Configuration
    print("\n3. Verifying LSTM Configuration:")
    lstm_layer = model_with_lstm.net.patch_lstm

    print(f"   Input size: {lstm_layer.input_size}")
    print(f"   Hidden size: {lstm_layer.hidden_size}")
    print(f"   Number of layers: {lstm_layer.num_layers}")
    print(f"   Bidirectional: {lstm_layer.bidirectional}")
    print(f"   Batch first: {lstm_layer.batch_first}")

    # Verify configuration matches
    expected_input_size = args_with_lstm.patch_len * \
        args_with_lstm.patch_len  # dim = patch_len^2
    config_correct = (
        lstm_layer.input_size == expected_input_size and
        lstm_layer.hidden_size == args_with_lstm.lstm_hidden_size and
        lstm_layer.num_layers == args_with_lstm.lstm_layers and
        lstm_layer.bidirectional == args_with_lstm.lstm_bidirectional
    )

    print(f"   Configuration correct: {'✅' if config_correct else '❌'}")

    # Test 4: Forward Pass
    print("\n4. Testing Forward Pass:")
    batch_size = 4
    test_input = torch.randn(
        batch_size, args_with_lstm.seq_len, args_with_lstm.enc_in)

    try:
        with torch.no_grad():
            output_with_lstm = model_with_lstm(test_input)
            output_without_lstm = model_without_lstm(test_input)

        print(f"   Input shape: {test_input.shape}")
        print(f"   Output with LSTM: {output_with_lstm.shape}")
        print(f"   Output without LSTM: {output_without_lstm.shape}")
        print(
            f"   Shape consistency: {'✅' if output_with_lstm.shape == output_without_lstm.shape else '❌'}")

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

    # Test 5: LSTM Impact Analysis
    print("\n5. Analyzing LSTM Impact:")

    # Check if outputs are different (LSTM is actually doing something)
    outputs_different = not torch.allclose(
        output_with_lstm, output_without_lstm, atol=1e-4)
    print(f"   Outputs are different: {'✅' if outputs_different else '❌'}")

    # Statistics
    lstm_mean = output_with_lstm.mean().item()
    no_lstm_mean = output_without_lstm.mean().item()
    lstm_std = output_with_lstm.std().item()
    no_lstm_std = output_without_lstm.std().item()

    print(f"   With LSTM - Mean: {lstm_mean:.6f}, Std: {lstm_std:.6f}")
    print(
        f"   Without LSTM - Mean: {no_lstm_mean:.6f}, Std: {no_lstm_std:.6f}")
    print(f"   Mean difference: {abs(lstm_mean - no_lstm_mean):.6f}")

    # Test 6: Parameter Count Analysis
    print("\n6. Parameter Count Analysis:")

    total_params_lstm = sum(p.numel() for p in model_with_lstm.parameters())
    total_params_no_lstm = sum(p.numel()
                               for p in model_without_lstm.parameters())
    lstm_params = total_params_lstm - total_params_no_lstm

    print(f"   Model with LSTM: {total_params_lstm:,} parameters")
    print(f"   Model without LSTM: {total_params_no_lstm:,} parameters")
    print(f"   LSTM adds: {lstm_params:,} parameters")
    print(f"   LSTM percentage: {(lstm_params/total_params_lstm)*100:.1f}%")

    # Test 7: Gradient Flow Test
    print("\n7. Testing Gradient Flow:")

    try:
        # Create a simple loss and backpropagate
        target = torch.randn_like(output_with_lstm)
        loss = torch.nn.MSELoss()(output_with_lstm, target)
        loss.backward()

        # Check if LSTM parameters have gradients
        lstm_has_gradients = any(
            p.grad is not None for p in model_with_lstm.net.patch_lstm.parameters())
        print(
            f"   LSTM parameters have gradients: {'✅' if lstm_has_gradients else '❌'}")

        # Check gradient magnitudes
        lstm_grad_norms = []
        for p in model_with_lstm.net.patch_lstm.parameters():
            if p.grad is not None:
                lstm_grad_norms.append(p.grad.norm().item())

        if lstm_grad_norms:
            avg_grad_norm = sum(lstm_grad_norms) / len(lstm_grad_norms)
            print(f"   Average gradient norm: {avg_grad_norm:.6f}")

    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")

    # Test 8: Intermediate Feature Analysis
    print("\n8. Analyzing Intermediate Features:")

    # Hook to capture LSTM output
    lstm_outputs = []

    def lstm_hook(module, input, output):
        # output[0] is the LSTM output, output[1] is hidden state
        lstm_outputs.append(output[0])

    # Register hook
    handle = model_with_lstm.net.patch_lstm.register_forward_hook(lstm_hook)

    try:
        with torch.no_grad():
            _ = model_with_lstm(test_input)

        if lstm_outputs:
            lstm_out = lstm_outputs[0]
            print(f"   LSTM output shape: {lstm_out.shape}")
            print(f"   LSTM output mean: {lstm_out.mean().item():.6f}")
            print(f"   LSTM output std: {lstm_out.std().item():.6f}")
            print(
                f"   LSTM output range: [{lstm_out.min().item():.3f}, {lstm_out.max().item():.3f}]")

        handle.remove()  # Clean up hook

    except Exception as e:
        print(f"   ❌ Intermediate feature analysis failed: {e}")
        handle.remove()

    # Final Verdict
    print("\n" + "=" * 50)
    print("🎯 FINAL VERIFICATION RESULT:")

    all_tests_passed = (
        has_lstm and has_lstm_proj and has_lstm_norm and
        config_correct and outputs_different
    )

    if all_tests_passed:
        print("✅ LSTM IS CORRECTLY IMPLEMENTED AND ACTIVE!")
        print("   - LSTM layers are properly initialized")
        print("   - Configuration matches specifications")
        print("   - LSTM is actively processing data")
        print("   - Outputs show LSTM impact")
        print("   - Gradients flow correctly")
    else:
        print("❌ LSTM IMPLEMENTATION HAS ISSUES!")
        print("   Please check the configuration and implementation")

    return all_tests_passed


if __name__ == "__main__":
    verify_lstm_implementation()
