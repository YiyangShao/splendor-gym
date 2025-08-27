#!/usr/bin/env python3
"""
Colab-Compatible Splendor AlphaZero Training Script

This version is designed to work out-of-the-box in Google Colab without 
requiring C++ compilation. It uses Python-only MCTS implementations.

Usage in Colab:
1. Clone the repo: !git clone https://github.com/your-repo/splendor-gym.git
2. Change directory: %cd splendor-gym  
3. Run training: !python train_splendor_colab.py

Or with custom settings:
!python train_splendor_colab.py --max_env_step 10000 --num_simulations 25
"""

import os
import sys
import argparse
import time
from pathlib import Path

def setup_colab_environment():
    """Setup the environment for Google Colab"""
    print("🔧 Setting up Google Colab environment...")
    
    # Add LightZero to Python path
    current_dir = os.getcwd()
    lightzero_path = os.path.join(current_dir, 'LightZero')
    
    if lightzero_path not in sys.path:
        sys.path.insert(0, lightzero_path)
        print(f"✅ Added {lightzero_path} to Python path")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        return True
    except ImportError:
        print("ℹ️ Not running in Google Colab (local environment)")
        return False

def fix_ctree_imports():
    """Fix ctree import issues by making them conditional"""
    print("🔧 Fixing ctree import issues...")
    
    # Patch the problematic import file
    mcts_ctree_file = 'LightZero/lzero/mcts/tree_search/mcts_ctree.py'
    
    if os.path.exists(mcts_ctree_file):
        with open(mcts_ctree_file, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'try:' in content and 'from lzero.mcts.ctree.ctree_efficientzero import ez_tree' in content:
            print("✅ mcts_ctree.py already patched")
            return True
        
        # Apply patch to make imports conditional
        old_imports = """from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero
from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero"""

        new_imports = """# Conditional imports for C++ extensions (with fallback for Colab)
try:
    from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
    CTREE_EFFICIENTZERO_AVAILABLE = True
except ImportError:
    tree_efficientzero = None
    CTREE_EFFICIENTZERO_AVAILABLE = False

try:
    from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero
    CTREE_GUMBEL_MUZERO_AVAILABLE = True
except ImportError:
    tree_gumbel_muzero = None
    CTREE_GUMBEL_MUZERO_AVAILABLE = False

try:
    from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero
    CTREE_MUZERO_AVAILABLE = True
except ImportError:
    tree_muzero = None
    CTREE_MUZERO_AVAILABLE = False"""

        if old_imports in content:
            new_content = content.replace(old_imports, new_imports)
            
            # Backup original file
            with open(mcts_ctree_file + '.backup', 'w') as f:
                f.write(content)
            
            # Write patched file
            with open(mcts_ctree_file, 'w') as f:
                f.write(new_content)
            
            print("✅ Applied ctree import patch")
            return True
        else:
            print("⚠️ mcts_ctree.py structure has changed, manual fix needed")
            return False
    else:
        print(f"❌ {mcts_ctree_file} not found")
        return False

def create_colab_config(args):
    """Create Colab-optimized config"""
    
    # Import after path setup
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    from easydict import EasyDict
    import copy
    
    # Start with base config
    config = EasyDict(copy.deepcopy(splendor_alphazero_config))
    create_config = EasyDict(copy.deepcopy(splendor_alphazero_create_config))
    
    # COLAB OPTIMIZATIONS:
    
    # 1. Force Python trees (no C++ compilation needed)
    config.policy.mcts_ctree = False
    print("✅ Using Python MCTS (no C++ compilation required)")
    
    # 2. Use 'base' environment manager (synchronous, reliable)
    config.env.manager.type = 'base'
    
    # 3. Optimize for Colab resources
    config.env.max_turns = args.max_turns
    config.policy.mcts.num_simulations = args.num_simulations
    
    # 4. Environment settings
    config.env.collector_env_num = args.collector_env_num
    config.env.evaluator_env_num = args.evaluator_env_num
    config.env.n_evaluator_episode = args.evaluator_env_num
    
    # 5. Policy settings
    config.policy.collector_env_num = args.collector_env_num
    config.policy.evaluator_env_num = args.evaluator_env_num
    config.policy.batch_size = args.batch_size
    config.policy.learning_rate = args.learning_rate
    config.policy.eval_freq = args.eval_freq
    
    # 6. Training settings
    config.exp_name = f'data_az_ctree/{args.exp_name}_seed{args.seed}'
    
    # 7. Use CPU by default (Colab CPU is often faster than GPU for small models)
    config.policy.device = args.device
    config.policy.cuda = (args.device != 'cpu')
    
    # 8. Colab-specific memory optimizations
    config.policy.batch_size = min(config.policy.batch_size, 128)  # Smaller batches for Colab
    
    return config, create_config

def estimate_colab_training_times(config):
    """Provide realistic time estimates for Colab"""
    
    max_turns = config.env.max_turns
    num_simulations = config.policy.mcts.num_simulations
    collector_envs = config.env.collector_env_num
    
    # Colab is typically slower than local machines
    colab_multiplier = 1.5
    
    # Estimate times (adjusted for Python MCTS + Colab)
    python_mcts_overhead = 1.3  # Python MCTS is ~30% slower than C++
    estimated_episode_turns = min(max_turns, 60)
    mcts_time_per_step = (0.01 + (num_simulations * 0.005)) * python_mcts_overhead * colab_multiplier
    episode_time = estimated_episode_turns * mcts_time_per_step
    collection_time_per_episode = episode_time + 3  # +overhead
    
    episodes_per_iteration = config.policy.get('collect_n_episode', 8)
    iteration_time = (episodes_per_iteration / collector_envs) * collection_time_per_episode
    learning_time_per_iteration = 1.0 * colab_multiplier  # Learning overhead
    total_iteration_time = iteration_time + learning_time_per_iteration
    
    print(f"\n📊 COLAB TRAINING TIME ESTIMATES:")
    print(f"=" * 50)
    print(f"Configuration:")
    print(f"  - Python MCTS (no C++ compilation)")
    print(f"  - Max turns: {max_turns}")
    print(f"  - MCTS simulations: {num_simulations}")
    print(f"  - Collector environments: {collector_envs}")
    print(f"")
    print(f"Time Estimates (adjusted for Colab):")
    print(f"  - MCTS forward pass: ~{mcts_time_per_step:.3f}s")
    print(f"  - Single episode: ~{collection_time_per_episode:.1f}s")
    print(f"  - Training iteration: ~{total_iteration_time:.1f}s")
    print(f"")
    print(f"Colab Training Estimates:")
    print(f"  - 50 iterations: ~{(total_iteration_time * 50 / 60):.1f} minutes")
    print(f"  - 200 iterations: ~{(total_iteration_time * 200 / 60):.1f} minutes")
    print(f"  - 1,000 iterations: ~{(total_iteration_time * 1000 / 3600):.1f} hours")
    print(f"")
    print(f"💡 Colab Tips:")
    print(f"  - Use CPU for small-scale training (often faster)")
    print(f"  - Keep sessions under 12 hours (Colab limit)")
    print(f"  - Save checkpoints regularly")
    print(f"=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Colab-Compatible Splendor AlphaZero Training')
    
    # Colab-optimized defaults
    parser.add_argument('--max_env_step', type=int, default=20000,
                        help='Maximum environment steps (default: 20K for Colab)')
    parser.add_argument('--exp_name', type=str, default='colab_splendor',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Conservative defaults for Colab
    parser.add_argument('--num_simulations', type=int, default=15,
                        help='MCTS simulations (15=fast for Colab, 25=medium, 50=slow)')
    parser.add_argument('--max_turns', type=int, default=40,
                        help='Max turns per game (40=medium, 80=long)')
    parser.add_argument('--collector_env_num', type=int, default=1,
                        help='Collector environments (keep at 1 for Colab)')
    parser.add_argument('--evaluator_env_num', type=int, default=1,
                        help='Evaluator environments (keep at 1 for Colab)')
    
    # Training parameters  
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (smaller for Colab)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--eval_freq', type=int, default=1000, help='Evaluation frequency')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device (cpu recommended for Colab)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 COLAB-COMPATIBLE SPLENDOR ALPHAZERO TRAINING")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Max env steps: {args.max_env_step:,}")
    print(f"MCTS simulations: {args.num_simulations}")
    print(f"Max turns per game: {args.max_turns}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    # Setup environment
    is_colab = setup_colab_environment()
    
    # Fix ctree imports
    if not fix_ctree_imports():
        print("⚠️ Could not patch ctree imports automatically")
        print("💡 Training may still work with Python MCTS fallback")
    
    # Create config
    try:
        config, create_config = create_colab_config(args)
        print("✅ Configuration created successfully")
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        print("💡 Make sure you're in the correct directory with LightZero/")
        return
    
    # Show estimates
    estimate_colab_training_times(config)
    
    print(f"\n🚀 Starting training...")
    print(f"Logs: {config.exp_name}")
    if is_colab:
        print(f"Monitor with: !tensorboard --logdir {config.exp_name}")
    else:
        print(f"Monitor with: tensorboard --logdir {config.exp_name}")
    
    try:
        start_time = time.time()
        
        # Import and start training
        from lzero.entry import train_alphazero
        
        train_alphazero(
            [config, create_config], 
            seed=args.seed, 
            max_env_step=args.max_env_step
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Training completed successfully in {elapsed:.1f}s!")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n⏹️ Training interrupted after {elapsed:.1f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Training failed after {elapsed:.1f}s: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Try: !pip install -e LightZero/")
        print("3. Check that all dependencies are installed")
        raise

if __name__ == '__main__':
    main()
