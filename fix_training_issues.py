#!/usr/bin/env python3
"""
Fix Training Issues Script

This script addresses the multiprocessing and subprocess environment manager issues
that are causing the AlphaZero training to fail silently.

Issues found:
1. Multiprocessing fork() deprecation warnings on macOS
2. Subprocess environment manager failures 
3. Training process terminating silently

Solutions:
1. Use 'spawn' start method instead of 'fork' on macOS
2. Switch to synchronous environment manager to avoid subprocess issues
3. Add proper error handling and logging
"""

import os
import sys
import multiprocessing as mp
from pathlib import Path

# Add LightZero to path
LIGHTZERO_PATH = Path(__file__).parent / 'LightZero'
sys.path.insert(0, str(LIGHTZERO_PATH))

def fix_multiprocessing():
    """Set multiprocessing start method to avoid fork() issues on macOS."""
    # Check if we're on macOS
    if sys.platform == 'darwin':
        try:
            # Set start method to 'spawn' to avoid fork() deprecation warnings
            mp.set_start_method('spawn', force=True)
            print("✓ Set multiprocessing start method to 'spawn' for macOS compatibility")
        except RuntimeError as e:
            print(f"Warning: Could not set multiprocessing start method: {e}")
    
def create_fixed_config():
    """Create a fixed configuration that avoids subprocess environment manager."""
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    from easydict import EasyDict
    import copy
    
    # Create fixed config
    fixed_config = EasyDict(copy.deepcopy(splendor_alphazero_config))
    fixed_create_config = EasyDict(copy.deepcopy(splendor_alphazero_create_config))
    
    # CRITICAL FIX: Use base environment manager instead of subprocess
    # This avoids the multiprocessing issues entirely
    fixed_create_config.env_manager = EasyDict({'type': 'base', 'episode_num': 1})  # Changed from 'subprocess'
    
    # Reduce complexity for initial testing
    fixed_config.env.collector_env_num = 1  # Start with single collector
    fixed_config.env.evaluator_env_num = 1  # Start with single evaluator
    fixed_config.policy.collector_env_num = 1
    fixed_config.policy.evaluator_env_num = 1
    fixed_config.policy.n_episode = 1
    
    # Add debugging and error handling
    fixed_config.policy.learn.learner.hook.log_show_after_iter = 1  # More frequent logging
    
    return fixed_config, fixed_create_config

def test_environment_creation():
    """Test if we can create environments successfully with the fixed config."""
    try:
        print("Testing environment creation...")
        
        from LightZero.zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
        
        # Test single environment creation
        env = SplendorLightZeroEnv()
        obs = env.reset()
        legal_actions = env.legal_actions
        print(f"✓ Single environment works, legal actions: {len(legal_actions)}")
        
        # Test with DI-engine environment manager
        from ding.envs import create_env_manager
        from functools import partial
        
        def env_fn(cfg=None):
            return SplendorLightZeroEnv(cfg)
        
        # Test base environment manager (our fix)
        print("Testing base environment manager...")
        from easydict import EasyDict
        base_env_manager = create_env_manager(
            EasyDict({'type': 'base', 'episode_num': 1}), 
            [partial(env_fn, cfg={}) for _ in range(1)]
        )
        base_env_manager.seed(0)
        print("✓ Base environment manager works")
        
        base_env_manager.close()
        print("✓ Environment test passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_fixed_training():
    """Run training with the fixed configuration."""
    print("Starting fixed AlphaZero training...")
    
    try:
        # Apply multiprocessing fix
        fix_multiprocessing()
        
        # Test environment first
        if not test_environment_creation():
            print("Environment test failed, aborting training")
            return False
        
        # Get fixed config
        config, create_config = create_fixed_config()
        
        # Set experiment name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        config.exp_name = f'data_az_ctree/fixed_training_seed0_{timestamp}'
        
        print(f"Experiment name: {config.exp_name}")
        print("Configuration fixes applied:")
        print(f"- Environment manager: {create_config.env_manager.type}")
        print(f"- Collector environments: {config.env.collector_env_num}")
        print(f"- Evaluator environments: {config.env.evaluator_env_num}")
        
        # Import and run training
        from lzero.entry import train_alphazero
        
        print("Starting training with fixed configuration...")
        train_alphazero([config, create_config], seed=0, max_env_step=10000)  # Short test run
        
        print("✓ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("="*60)
    print("SPLENDOR ALPHAZERO TRAINING FIX")
    print("="*60)
    print("Applying fixes for:")
    print("1. Multiprocessing fork() deprecation warnings")
    print("2. Subprocess environment manager failures")
    print("3. Silent training termination")
    print("="*60)
    
    success = run_fixed_training()
    
    print("="*60)
    if success:
        print("✓ ALL FIXES APPLIED SUCCESSFULLY")
        print("Training should now work without buffer zero issues!")
    else:
        print("✗ FIXES FAILED")
        print("Please check the error messages above.")
    print("="*60)
