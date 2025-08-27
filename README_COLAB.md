# 🚀 Running Splendor AlphaZero in Google Colab

This guide explains how to run Splendor AlphaZero training in Google Colab without any manual setup or compilation.

## 🔥 Quick Start (Clone and Run)

```bash
# 1. Clone the repository
!git clone https://github.com/your-username/splendor-gym.git

# 2. Change to the project directory  
%cd splendor-gym

# 3. Run automatic setup (installs dependencies, patches imports)
!python colab_setup.py

# 4. Start training immediately
!python train_splendor_colab.py
```

That's it! Training will start automatically with Colab-optimized settings.

## 📊 What You'll See

The training will show:
- ✅ Environment episodes completing (~3-10s each)
- ✅ Buffer receiving data (transitions flowing) 
- ✅ Model learning (loss values updating)
- ✅ Checkpoints being saved

## ⚙️ Custom Training Settings

```bash
# Fast training (for testing)
!python train_splendor_colab.py --max_env_step 5000 --num_simulations 10 --max_turns 30

# Medium quality training
!python train_splendor_colab.py --max_env_step 50000 --num_simulations 25 --max_turns 50

# High quality training (long run)
!python train_splendor_colab.py --max_env_step 200000 --num_simulations 40 --max_turns 80
```

## 📈 Monitoring Training

```bash
# Start TensorBoard
!tensorboard --logdir data_az_ctree/ --host 0.0.0.0 --port 6006

# View in Colab
from google.colab import output
output.serve_kernel_port_as_iframe(6006)
```

## ⏱️ Time Estimates for Colab

| Configuration | Episode Time | Iteration Time | 50 Iterations | 200 Iterations |
|---------------|-------------|----------------|---------------|-----------------|
| **Fast** (10 MCTS, 30 turns) | ~4s | ~20s | 17 min | 67 min |
| **Medium** (25 MCTS, 50 turns) | ~8s | ~35s | 29 min | 117 min |
| **High** (40 MCTS, 80 turns) | ~15s | ~60s | 50 min | 200 min |

*Times are estimates for Colab CPU. GPU may be faster for larger models.*

## 🔧 What Makes This Colab-Compatible?

### 1. **No C++ Compilation Required**
- Uses Python MCTS implementation (`ptree`) instead of C++ (`ctree`)
- Automatically patches problematic imports
- Works out-of-the-box on any system

### 2. **Automatic Environment Setup**
- Installs all required dependencies
- Configures Python paths correctly
- Handles import issues gracefully

### 3. **Optimized for Colab Resources**
- Conservative default settings (smaller models, fewer simulations)
- Memory-efficient batch sizes
- CPU-optimized by default

### 4. **Robust Error Handling**
- Graceful fallbacks for missing components
- Clear error messages and troubleshooting tips
- Continues training even if some features unavailable

## 🐍 Key Technical Details

### Python vs C++ MCTS
```python
# The training automatically uses:
config.policy.mcts_ctree = False  # Python implementation

# Instead of:
config.policy.mcts_ctree = True   # C++ implementation (requires compilation)
```

### Import Patching
The setup automatically patches `mcts_ctree.py` to handle missing C++ extensions:
```python
# Before (breaks in Colab):
from lzero.mcts.ctree.ctree_efficientzero import ez_tree

# After (Colab-compatible):
try:
    from lzero.mcts.ctree.ctree_efficientzero import ez_tree
except ImportError:
    ez_tree = None  # Use Python fallback
```

## 🚨 Troubleshooting

### "ImportError: cannot import name 'ez_tree'"
```bash
# Re-run the setup script
!python colab_setup.py

# Or manually patch:
!python fix_colab_ctree.py
```

### "No module named 'lzero'"
```bash
# Make sure you're in the correct directory
%cd splendor-gym

# Verify LightZero is present
!ls -la LightZero/
```

### Training seems slow
- Use CPU instead of GPU for small models: `--device cpu`
- Reduce MCTS simulations: `--num_simulations 15`
- Reduce max turns: `--max_turns 30`

### Out of memory
```bash
# Use smaller batch size
!python train_splendor_colab.py --batch_size 64

# Or reduce environment count
!python train_splendor_colab.py --collector_env_num 1
```

## 🎯 Training Quality vs Speed

| Priority | MCTS Sims | Max Turns | Time/Iteration | Quality |
|----------|-----------|-----------|----------------|---------|
| **Speed** | 10 | 30 | ~20s | Basic |
| **Balanced** | 25 | 50 | ~35s | Good |
| **Quality** | 40 | 80 | ~60s | High |
| **Research** | 100+ | 100+ | ~180s+ | Maximum |

## 💡 Colab Pro Tips

1. **Save checkpoints regularly** - Colab sessions can disconnect
2. **Use CPU for initial testing** - Often faster than GPU for small models
3. **Monitor resource usage** - Keep an eye on RAM and disk usage
4. **Download checkpoints** - Save important models to Google Drive

## 🔗 Files Overview

- `train_splendor_colab.py` - Main Colab-compatible training script
- `colab_setup.py` - Automatic environment setup
- `fix_colab_ctree.py` - Manual ctree import fix (if needed)
- `README_COLAB.md` - This guide

## 🎉 Success Indicators

You'll know everything is working when you see:
```
✅ Episodes complete properly (turn limit enforced)
✅ Buffer receives data (1000+ transitions collected)
✅ Model learning (loss values decreasing)
✅ Checkpoints saving regularly
```

Happy training! 🚀
