"""
Preflight check for sparsify SAE training on HPC.
Run this on the login node (no GPU needed) before allocating A100 time.

Usage: python preflight_check.py
"""

import sys
import os
import shutil

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

failed = False


def check(label, condition, msg_fail="", msg_pass=""):
    global failed
    if condition:
        print(f"  {PASS} {label}" + (f" — {msg_pass}" if msg_pass else ""))
    else:
        print(f"  {FAIL} {label}" + (f" — {msg_fail}" if msg_fail else ""))
        failed = True


def warn(label, condition, msg_warn=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {WARN} {label}" + (f" — {msg_warn}" if msg_warn else ""))


# ── 1. Python version ──
print("\n1. Python version")
v = sys.version_info
check(
    f"Python {v.major}.{v.minor}.{v.micro}",
    v >= (3, 10),
    msg_fail="sparsify requires Python >= 3.10",
)

# ── 2. Required packages ──
print("\n2. Required packages")
required_packages = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "safetensors": "safetensors",
    "einops": "einops",
    "simple_parsing": "simple-parsing",
    "natsort": "natsort",
    "schedulefree": "schedulefree",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface-hub",
    "tqdm": "tqdm",
}

for import_name, pip_name in required_packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        check(f"{pip_name} ({version})", True)
    except ImportError:
        check(f"{pip_name}", False, msg_fail=f"pip install {pip_name}")

# Optional but used at runtime
print("\n   Optional packages:")
for import_name, pip_name in [("triton", "triton"), ("wandb", "wandb"), ("bitsandbytes", "bitsandbytes")]:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        print(f"  {PASS} {pip_name} ({version})")
    except ImportError:
        warn(f"{pip_name}", False, msg_warn=f"not installed (optional, pip install {pip_name})")

# ── 3. Sparsify importable ──
print("\n3. Sparsify package")
try:
    from sparsify import Sae, SaeConfig, Trainer, TrainConfig
    check("sparsify imports work", True)
except Exception as e:
    check("sparsify imports work", False, msg_fail=str(e))

# ── 4. CUDA / GPU ──
print("\n4. CUDA availability")
try:
    import torch
    check(f"PyTorch CUDA compiled: {torch.cuda.is_available()}", torch.cuda.is_available(),
          msg_fail="torch.cuda.is_available() is False — expected on login node, OK if GPUs exist on compute nodes")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f"  {PASS} GPU {i}: {name} ({mem:.1f} GB)")
    else:
        print(f"  {WARN} No GPU visible (expected on login node — will be available on compute node)")
except Exception as e:
    print(f"  {WARN} Could not check CUDA: {e}")

# ── 5. Model access ──
print("\n5. Model access (Qwen/Qwen2.5-Math-1.5B)")
try:
    from huggingface_hub import model_info
    info = model_info("Qwen/Qwen2.5-Math-1.5B")
    check(f"Model accessible on Hub", True, msg_pass=f"id={info.id}")
except Exception as e:
    check("Model accessible on Hub", False, msg_fail=str(e))

# Check if model is already cached
try:
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache("Qwen/Qwen2.5-Math-1.5B", "config.json")
    if cached and cached is not None and not isinstance(cached, type(None)):
        print(f"  {PASS} Model already cached locally")
    else:
        print(f"  {WARN} Model not cached — will download on first run (~3GB). Consider pre-downloading:")
        print(f"        python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Math-1.5B')\"")
except Exception:
    pass

# ── 6. Dataset ──
print("\n6. Preprocessed dataset")
DATA_PATH = "/scratch4/workspace/rbhirud_umass_edu-RL_experiments/colm/data/math_plus_text"
if os.path.isdir(DATA_PATH):
    # Verify it loads
    try:
        from datasets import Dataset
        ds = Dataset.load_from_disk(DATA_PATH)
        check(f"Dataset loads from {DATA_PATH}", True, msg_pass=f"{len(ds)} examples, columns={ds.column_names}")
        check("Has 'text' column", "text" in ds.column_names, msg_fail=f"columns are {ds.column_names}, expected 'text'")
        # Spot check
        sample = ds[0]["text"]
        check("Sample text looks valid", len(sample) > 50, msg_pass=f"len={len(sample)}, starts with: {sample[:80]}...")
    except Exception as e:
        check(f"Dataset loads", False, msg_fail=str(e))
else:
    check(f"Dataset directory exists at {DATA_PATH}", False,
          msg_fail="Run prep_dataset.py first")

# ── 7. Save directory ──
print("\n7. Output directory")
SAVE_DIR = "/scratch4/workspace/rbhirud_umass_edu-RL_experiments/colm/sparsify/checkpoints"
parent = os.path.dirname(SAVE_DIR)
if os.path.isdir(parent):
    check(f"Parent dir exists: {parent}", True)
    # Test write permission
    test_file = os.path.join(parent, ".preflight_write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        check("Write permission OK", True)
    except OSError as e:
        check("Write permission", False, msg_fail=str(e))
else:
    print(f"  {WARN} Parent dir {parent} not found (expected — this is a local machine check)")

# ── 8. Disk space ──
print("\n8. Disk space")
for path in [DATA_PATH, "/scratch4/workspace/rbhirud_umass_edu-RL_experiments/colm"]:
    if os.path.exists(path):
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1e9
        check(f"{path}: {free_gb:.1f} GB free", free_gb > 10,
              msg_fail=f"Only {free_gb:.1f} GB free, recommend >10 GB for checkpoints")
        break
else:
    print(f"  {WARN} Cannot check disk space (paths not mounted locally)")

# ── 9. Dry-run CLI parse ──
print("\n9. CLI argument parsing (dry run)")
try:
    from simple_parsing import parse
    from sparsify.__main__ import RunConfig
    # Simulate the args that would be passed
    test_args = [
        "Qwen/Qwen2.5-Math-1.5B",
        DATA_PATH,
        "--batch_size", "16",
        "--save_every", "1000",
    ]
    old_argv = sys.argv
    sys.argv = ["sparsify"] + test_args
    try:
        cfg = parse(RunConfig)
        check("CLI args parse OK", True, msg_pass=f"model={cfg.model}, batch_size={cfg.batch_size}")
    finally:
        sys.argv = old_argv
except Exception as e:
    check("CLI args parse", False, msg_fail=str(e))

# ── Summary ──
print("\n" + "=" * 60)
if failed:
    print(f"{FAIL} Some checks failed. Fix the issues above before allocating GPU time.")
    sys.exit(1)
else:
    print(f"{PASS} All checks passed! Safe to allocate A100 and run training.")
    sys.exit(0)
