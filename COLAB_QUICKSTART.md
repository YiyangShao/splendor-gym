# 🚀 Google Colab Quick Start - Splendor AlphaZero

**TWO-COMMAND FIX for Python 3.12 and ctree import errors:**

```bash
# Fix 1: Python 3.12 compatibility (fixes lib2to3 errors)
!python fix_python312_compatibility.py

# Fix 2: ctree imports (fixes "cannot import name 'ez_tree'")  
!python fix_ctree_imports.py

# Then run training normally:
!python train_splendor_working.py --max_env_step 1000 --num_simulations 10
```

## 📋 Complete Colab Setup (6 Commands)

```bash
# 1. Clone the repository
!git clone https://github.com/your-repo/splendor-gym.git

# 2. Change directory
%cd splendor-gym

# 3. Fix Python 3.12 compatibility (ONE TIME ONLY)
!python fix_python312_compatibility.py

# 4. Fix import issues (ONE TIME ONLY)  
!python fix_ctree_imports.py

# 5. Fix Colab buffer issues (ONE TIME ONLY)
!python fix_colab_buffer_issues.py

# 6. Start training
!python train_colab_optimized.py --max_env_step 5000 --num_simulations 15
```

## 🔧 What the Fix Does

The `fix_ctree_imports.py` script:

1. **Patches the problematic import** in `mcts_ctree.py`:
   ```python
   # Before (breaks in Colab):
   from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
   
   # After (Colab-compatible):
   try:
       from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
   except ImportError:
       tree_efficientzero = None  # Use Python fallback
   ```

2. **Creates a safe `__init__.py`** for the ctree module
3. **Verifies the fix works** by testing imports
4. **Preserves original files** with `.original` backup

## ⚡ Training Options

### Quick Test (2-3 minutes)
```bash
!python train_splendor_working.py --max_env_step 500 --num_simulations 5 --max_turns 20
```

### Medium Training (15-20 minutes) 
```bash
!python train_splendor_working.py --max_env_step 10000 --num_simulations 15 --max_turns 40
```

### Quality Training (1-2 hours)
```bash
!python train_splendor_working.py --max_env_step 50000 --num_simulations 25 --max_turns 60
```

## 📊 Monitor Training

```bash
# Start TensorBoard (in a new cell)
%load_ext tensorboard
%tensorboard --logdir data_az_ctree/

# Or command line version
!tensorboard --logdir data_az_ctree/ --host 0.0.0.0 --port 6006
```

## 🔍 What Success Looks Like

You'll see:
```
✅ Episodes complete properly (turn limit enforced)
✅ Buffer receives data (1000+ transitions collected)  
✅ Model learning (loss values decreasing)
✅ Checkpoints saving regularly
```

## 🚨 Troubleshooting

### Still getting import errors?
```bash
# Fix Python 3.12 compatibility first
!python fix_python312_compatibility.py

# Then fix ctree imports
!python fix_ctree_imports.py

# Check if LightZero is present
!ls -la LightZero/

# Make sure you're in the right directory
%cd splendor-gym
```

### Getting "ModuleNotFoundError: No module named 'lib2to3'"?
```bash
# This is a Python 3.12 compatibility issue - run this fix:
!python fix_python312_compatibility.py
```

### Buffer statistics showing all zeros?
```bash
# Apply Colab-specific buffer fixes:
!python fix_colab_buffer_issues.py

# Run comprehensive diagnostics:
!python diagnose_colab_buffer.py

# If diagnostics pass, try minimal training:
!python test_colab_minimal_training.py
```

### Training too slow?
```bash
# Use fewer MCTS simulations
!python train_splendor_working.py --num_simulations 10

# Use shorter games
!python train_splendor_working.py --max_turns 30

# Use CPU (sometimes faster in Colab)
!python train_splendor_working.py --device cpu
```

### Out of memory?
```bash
# Smaller batch size
!python train_splendor_working.py --batch_size 64

# Single environment
!python train_splendor_working.py --collector_env_num 1
```

## 📁 Key Files

**Core Fixes:**
- `fix_python312_compatibility.py` - **PYTHON 3.12 FIX** for lib2to3 errors
- `fix_ctree_imports.py` - **CTREE FIX** for import errors
- `fix_colab_buffer_issues.py` - **BUFFER FIX** for Colab zero statistics

**Training Scripts:**
- `train_splendor_working.py` - Main training script (works after fixes)
- `train_colab_optimized.py` - Colab-optimized script (created by buffer fix)

**Diagnostic Tools (if issues persist):**
- `diagnose_colab_buffer.py` - Comprehensive component testing
- `test_colab_minimal_training.py` - Minimal training with monitoring

## 🎯 Success Checklist

- [ ] Fixed Python 3.12 compatibility with `fix_python312_compatibility.py`
- [ ] Fixed ctree imports with `fix_ctree_imports.py`
- [ ] Training starts without import errors
- [ ] Episodes complete (not hanging)
- [ ] Buffer shows non-zero statistics  
- [ ] Loss values are updating
- [ ] Checkpoints are being saved

## 💡 Pro Tips

1. **Save checkpoints to Google Drive** for persistence
2. **Use CPU for small models** (often faster than GPU in Colab)
3. **Start with small settings** to verify everything works
4. **Monitor RAM usage** to avoid out-of-memory errors

---

**🎉 With this fix, Splendor AlphaZero training should work perfectly in Google Colab!**
