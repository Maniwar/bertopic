#!/usr/bin/env python3
"""
Diagnostic script to check Flash Attention 2 installation and compatibility.
Run this in your actual runtime environment (where streamlit runs).
"""

import sys

print("=" * 60)
print("FLASH ATTENTION 2 DIAGNOSTIC TOOL")
print("=" * 60)

# 1. Check PyTorch
print("\n1. PyTorch Installation:")
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")

        # Check compute capability
        capability = torch.cuda.get_device_capability(0)
        compute_capability = capability[0] + capability[1] / 10
        print(f"   ℹ️  Compute Capability: {compute_capability}")

        if compute_capability >= 8.0:
            print(f"   ✅ GPU supports Flash Attention 2 (needs 8.0+)")
        else:
            print(f"   ❌ GPU does NOT support Flash Attention 2 (needs 8.0+, you have {compute_capability})")
            print(f"      Your GPU: {torch.cuda.get_device_name(0)}")
            print(f"      Flash Attention 2 requires: RTX 3000+, A100, H100, etc.")
    else:
        print(f"   ❌ CUDA not available - Flash Attention 2 requires GPU")
except ImportError:
    print("   ❌ PyTorch not installed")
    sys.exit(1)

# 2. Check Flash Attention
print("\n2. Flash Attention Installation:")
try:
    import flash_attn
    print(f"   ✅ flash-attn installed: version {flash_attn.__version__}")
except ImportError as e:
    print(f"   ❌ flash-attn not installed or import failed")
    print(f"      Error: {e}")

# 3. Check Transformers
print("\n3. Transformers Library:")
try:
    import transformers
    print(f"   ✅ Transformers version: {transformers.__version__}")

    # Check if transformers supports flash attention 2
    version_parts = transformers.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    if major >= 4 and minor >= 34:
        print(f"   ✅ Transformers supports Flash Attention 2 (needs 4.34+)")
    else:
        print(f"   ❌ Transformers too old (needs 4.34+, you have {transformers.__version__})")
except ImportError:
    print("   ❌ Transformers not installed")

# 4. Test Flash Attention 2
print("\n4. Testing Flash Attention 2 Loading:")
if torch.cuda.is_available():
    try:
        from transformers import AutoModelForCausalLM

        test_model = "microsoft/Phi-3-mini-4k-instruct"
        print(f"   Testing with: {test_model}")

        try:
            print("   Trying flash_attention_2...")
            model = AutoModelForCausalLM.from_pretrained(
                test_model,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
            print("   ✅ Flash Attention 2 WORKS!")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Flash Attention 2 failed: {e}")

            # Try SDPA
            try:
                print("   Trying sdpa...")
                model = AutoModelForCausalLM.from_pretrained(
                    test_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    attn_implementation="sdpa",
                    trust_remote_code=True
                )
                print("   ✅ SDPA works (fallback)")
                del model
                torch.cuda.empty_cache()
            except Exception as e2:
                print(f"   ❌ SDPA also failed: {e2}")

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
else:
    print("   ⚠️  Skipping test (no CUDA)")

# 5. Installation recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if not torch.cuda.is_available():
    print("❌ No CUDA - Flash Attention 2 requires GPU")
elif compute_capability < 8.0:
    print(f"❌ GPU too old (Compute {compute_capability}) - Use SDPA instead")
    print("   SDPA still gives 1.5-2x speedup on your GPU!")
else:
    print("\nTo install Flash Attention 2, try ONE of these methods:\n")

    print("METHOD 1 (Recommended - precompiled wheel):")
    print("  pip install flash-attn --no-build-isolation")

    print("\nMETHOD 2 (If method 1 fails - build from source):")
    print("  pip install ninja")
    print("  pip install flash-attn --no-build-isolation")

    print("\nMETHOD 3 (Specific CUDA version):")
    cuda_version = torch.version.cuda.replace('.', '')[:3]  # e.g., "121" for 12.1
    print(f"  # For your CUDA {torch.version.cuda}:")
    print(f"  pip install flash-attn --no-build-isolation")

    print("\nMETHOD 4 (If all else fails - use conda):")
    print("  conda install -c conda-forge flash-attn")

print("\n" + "=" * 60)
print("If Flash Attention 2 doesn't work, SDPA is a great fallback!")
print("SDPA gives 1.5-2x speedup with zero installation hassle.")
print("=" * 60)
