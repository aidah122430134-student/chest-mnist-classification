import torch

print("="*60)
print("GPU Configuration Test")
print("="*60)

# Check CUDA availability
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"2. CUDA Version: {torch.version.cuda}")
    print(f"3. cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"4. Device Count: {torch.cuda.device_count()}")
    print(f"5. Current Device: {torch.cuda.current_device()}")
    print(f"6. Device Name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor on GPU
    print(f"\n7. Testing tensor creation on GPU...")
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"   ✓ Matrix multiplication on GPU successful!")
    print(f"   Result tensor is on: {z.device}")
    
    # Memory usage
    print(f"\n8. GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("   CUDA is not available. Using CPU.")

print("\n" + "="*60)
print("Test completed!")
print("="*60)
