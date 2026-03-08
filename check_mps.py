#!/usr/bin/env python3
"""
Check MPS (Metal Performance Shaders) availability for PyTorch on Apple Silicon.
"""

import torch


def check_mps():
    print("=" * 50)
    print("PyTorch MPS (Metal Performance Shaders) Check")
    print("=" * 50)

    # Basic info
    print(f"\nPyTorch version: {torch.__version__}")

    # MPS availability checks
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()

    print(f"MPS built into PyTorch: {mps_built}")
    print(f"MPS device available: {mps_available}")

    if not mps_available:
        print("\n❌ MPS is NOT available.")
        if not mps_built:
            print("   Reason: PyTorch was not built with MPS support.")
        else:
            print("   Reason: MPS device not found (requires macOS 12.3+).")
        return False

    # Test MPS with a simple tensor operation
    print("\n" + "-" * 50)
    print("Running MPS acceleration test...")
    print("-" * 50)

    try:
        # Create tensors on MPS device
        device = torch.device("mps")

        # Simple matrix multiplication test
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warm-up
        _ = torch.matmul(a, b)
        torch.mps.synchronize()

        # Timed operation
        import time
        start = time.perf_counter()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        print(f"✓ Successfully created tensors on MPS device")
        print(f"✓ Matrix multiplication ({size}x{size}) x 10: {elapsed:.4f}s")
        print(f"✓ Result tensor shape: {c.shape}")
        print(f"✓ Result tensor device: {c.device}")

        print("\n" + "=" * 50)
        print("✅ MPS ACCELERATION IS AVAILABLE AND WORKING!")
        print("=" * 50)
        print("\nUsage example:")
        print('  device = torch.device("mps")')
        print('  tensor = torch.randn(100, 100, device=device)')
        print('  model = model.to(device)')

        return True

    except Exception as e:
        print(f"\n❌ MPS test failed with error: {e}")
        return False


if __name__ == "__main__":
    check_mps()
