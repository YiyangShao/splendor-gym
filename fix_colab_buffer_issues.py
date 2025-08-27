#!/usr/bin/env python3
"""
Fix Colab buffer issues

This script addresses common issues that cause buffer statistics to show zeros
in Google Colab environments.
"""

import sys
import os
import multiprocessing as mp
import traceback

def fix_multiprocessing_for_colab():
    """Fix multiprocessing issues in Colab"""
    print("🔧 Fixing multiprocessing for Colab...")
    
    try:
        # Force spawn method for Colab compatibility
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("✅ Set multiprocessing to 'spawn' method")
        else:
            print("✅ Multiprocessing already set to 'spawn'")
            
        return True
    except Exception as e:
        print(f"⚠️ Could not set multiprocessing method: {e}")
        return False

def apply_colab_environment_patches():
    """Apply patches specific to Colab environment"""
    print("🔧 Applying Colab environment patches...")
    
    # Patch 1: Ensure base environment manager is used
    config_file = "LightZero/zoo/board_games/splendor/config/splendor_alphazero_sp_mode_config.py"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check if env_manager is set to base
            if "env_manager=dict(type='subprocess')" in content:
                print("🔧 Switching from subprocess to base environment manager...")
                content = content.replace(
                    "env_manager=dict(type='subprocess')",
                    "env_manager=dict(type='base')"
                )
                
                with open(config_file, 'w') as f:
                    f.write(content)
                
                print("✅ Updated environment manager to 'base' for Colab compatibility")
            else:
                print("✅ Environment manager already set correctly")
                
        except Exception as e:
            print(f"⚠️ Could not patch config file: {e}")
    
    # Patch 2: Ensure Python MCTS is used
    try:
        # Check the config to ensure mcts_ctree = False
        from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
        
        if hasattr(splendor_alphazero_config.policy, 'mcts_ctree'):
            if splendor_alphazero_config.policy.mcts_ctree:
                print("⚠️ MCTS ctree is enabled, but may not work in Colab")
            else:
                print("✅ Python MCTS is enabled (good for Colab)")
        
    except Exception as e:
        print(f"⚠️ Could not check MCTS configuration: {e}")

def create_colab_optimized_config():
    """Create a Colab-optimized configuration file"""
    print("🔧 Creating Colab-optimized configuration...")
    
    config_content = '''#!/usr/bin/env python3
"""
Colab-optimized Splendor AlphaZero configuration

This configuration is specifically tuned for Google Colab environments
to ensure buffer data collection works properly.
"""

from easydict import EasyDict
from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
    splendor_alphazero_config as base_config,
    splendor_alphazero_create_config as base_create_config
)

# Create Colab-optimized config
colab_config = base_config.copy()
colab_create_config = base_create_config.copy()

# Critical fixes for Colab buffer collection
colab_config.env_manager = dict(type='base')  # Avoid subprocess issues
colab_config.policy.mcts_ctree = False  # Use Python MCTS only
colab_config.collector_env_num = 1  # Single environment for stability
colab_config.evaluator_env_num = 1

# Optimized settings for Colab constraints
colab_config.policy.num_simulations = 10  # Reasonable for Colab
colab_config.env.max_turns = 30  # Prevent infinite games
colab_config.replay_buffer.replay_buffer_size = 2000  # Moderate size
colab_config.batch_size = 16  # Reasonable batch size

# Ensure stable learning
colab_config.learn.learner.hook.save_ckpt_after_iter = 50
colab_config.learn.learner.log_show_after_iter = 10

if __name__ == '__main__':
    # Test the config
    print("Testing Colab-optimized config...")
    print(f"Environment manager: {colab_config.env_manager}")
    print(f"MCTS ctree: {colab_config.policy.mcts_ctree}")
    print(f"Collector envs: {colab_config.collector_env_num}")
    print("✅ Colab config looks good!")
'''
    
    try:
        with open('colab_optimized_config.py', 'w') as f:
            f.write(config_content)
        print("✅ Created colab_optimized_config.py")
        return True
    except Exception as e:
        print(f"❌ Could not create optimized config: {e}")
        return False

def create_colab_training_script():
    """Create a Colab-specific training script"""
    print("🔧 Creating Colab training script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Colab-specific Splendor AlphaZero training script

This script is optimized for Google Colab and includes all necessary fixes
for buffer data collection.
"""

import sys
import os
import time
import multiprocessing as mp

def setup_colab_environment():
    """Setup environment for Colab"""
    print("🔧 Setting up Colab environment...")
    
    # Apply compatibility fixes
    os.system('python fix_python312_compatibility.py')
    os.system('python fix_ctree_imports.py')
    
    # Add paths
    sys.path.insert(0, 'LightZero')
    sys.path.insert(0, '.')
    
    # Fix multiprocessing
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

def train_colab_optimized(max_env_step=1000, num_simulations=10, exp_name="colab_training"):
    """Train with Colab-optimized settings"""
    
    setup_colab_environment()
    
    print(f"🚀 Starting Colab-optimized training...")
    print(f"Experiment: {exp_name}")
    print(f"Max env steps: {max_env_step}")
    print(f"MCTS simulations: {num_simulations}")
    
    try:
        from lzero.entry import train_alphazero
        from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
            splendor_alphazero_config, splendor_alphazero_create_config
        )
        
        # Apply Colab-specific settings
        config = splendor_alphazero_config.copy()
        create_config = splendor_alphazero_create_config.copy()
        
        # Critical Colab fixes
        config.env_manager = dict(type='base')  # Avoid subprocess
        config.policy.mcts_ctree = False  # Python MCTS only
        config.collector_env_num = 1  # Single environment
        config.evaluator_env_num = 1
        
        # User settings
        config.max_env_step = max_env_step
        config.policy.num_simulations = num_simulations
        config.exp_name = exp_name
        
        # Buffer monitoring settings
        config.replay_buffer.replay_buffer_size = max(500, max_env_step // 2)
        
        print(f"📊 Buffer will collect data from {config.collector_env_num} environment(s)")
        print(f"⚙️ Using {config.policy.num_simulations} MCTS simulations per move")
        
        # Start training
        start_time = time.time()
        train_alphazero([config, create_config], seed=0)
        
        elapsed = time.time() - start_time
        print(f"\\n✅ Training completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Colab-optimized Splendor AlphaZero training')
    parser.add_argument('--max_env_step', type=int, default=1000, help='Maximum environment steps')
    parser.add_argument('--num_simulations', type=int, default=10, help='MCTS simulations per move')
    parser.add_argument('--exp_name', type=str, default='colab_training', help='Experiment name')
    
    args = parser.parse_args()
    
    train_colab_optimized(
        max_env_step=args.max_env_step,
        num_simulations=args.num_simulations,
        exp_name=args.exp_name
    )
'''
    
    try:
        with open('train_colab_optimized.py', 'w') as f:
            f.write(script_content)
        print("✅ Created train_colab_optimized.py")
        return True
    except Exception as e:
        print(f"❌ Could not create training script: {e}")
        return False

def check_colab_memory():
    """Check available memory in Colab"""
    print("🔍 Checking Colab memory...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        print(f"Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        if available_gb < 2.0:
            print("⚠️ Low memory detected - consider reducing batch size")
            return False
        else:
            print("✅ Sufficient memory available")
            return True
            
    except ImportError:
        print("psutil not available - cannot check memory")
        return True
    except Exception as e:
        print(f"⚠️ Memory check failed: {e}")
        return True

def verify_colab_fixes():
    """Verify all fixes are working"""
    print("🔍 Verifying Colab fixes...")
    
    try:
        # Test basic imports
        sys.path.insert(0, 'LightZero')
        sys.path.insert(0, '.')
        
        from lzero import __version__ as lzero_version
        print(f"✅ LightZero import successful (v{lzero_version})")
        
        from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
        print(f"✅ Config import successful")
        print(f"   Environment manager: {splendor_alphazero_config.env_manager}")
        print(f"   MCTS ctree: {splendor_alphazero_config.policy.mcts_ctree}")
        
        from zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
        env = SplendorLightZeroEnv()
        obs = env.reset()
        print(f"✅ Environment test successful")
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main fix function"""
    print("🩺 FIXING COLAB BUFFER ISSUES")
    print("=" * 50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("ℹ️ Not in Google Colab (running locally)")
    
    # Apply fixes
    success = True
    
    # Fix 1: Multiprocessing
    if not fix_multiprocessing_for_colab():
        success = False
    
    # Fix 2: Environment patches
    apply_colab_environment_patches()
    
    # Fix 3: Memory check
    if not check_colab_memory():
        print("⚠️ Consider using smaller model/batch sizes")
    
    # Fix 4: Create optimized components
    if not create_colab_optimized_config():
        success = False
    
    if not create_colab_training_script():
        success = False
    
    # Fix 5: Verify everything works
    if not verify_colab_fixes():
        success = False
    
    if success:
        print("\n🎉 COLAB BUFFER FIXES APPLIED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run: python diagnose_colab_buffer.py")
        print("2. If diagnostics pass, run: python train_colab_optimized.py")
        print("3. Monitor buffer statistics in the output")
        print("\nThe buffer should show non-zero 'pushed_in' values within 2-3 minutes")
    else:
        print("\n⚠️ Some fixes failed, but training might still work")
        print("Try running the diagnostic script to identify remaining issues")
    
    return success

if __name__ == '__main__':
    main()
