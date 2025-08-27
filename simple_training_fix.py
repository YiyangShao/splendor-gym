#!/usr/bin/env python3
"""
Simple Training Fix for Splendor AlphaZero

This is a simplified approach that just fixes the multiprocessing issue
and tests training with minimal changes to the existing configuration.
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
    if sys.platform == 'darwin':
        try:
            mp.set_start_method('spawn', force=True)
            print("✓ Set multiprocessing start method to 'spawn' for macOS compatibility")
        except RuntimeError as e:
            print(f"Warning: Could not set multiprocessing start method: {e}")

def run_simple_training():
    """Run training with minimal fixes - just multiprocessing and reduced scale."""
    print("Starting simple fixed AlphaZero training...")
    
    try:
        # Apply multiprocessing fix
        fix_multiprocessing()
        
        # Import configurations
        from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
            splendor_alphazero_config, splendor_alphazero_create_config
        )
        from easydict import EasyDict
        import copy
        
        # Create minimal modified config
        config = EasyDict(copy.deepcopy(splendor_alphazero_config))
        create_config = EasyDict(copy.deepcopy(splendor_alphazero_create_config))
        
        # Only change what's absolutely necessary
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        config.exp_name = f'data_az_ctree/simple_fix_seed0_{timestamp}'
        
        # Reduce scale for testing - keep original env manager
        config.env.collector_env_num = 2  # Reduce from 8 to 2
        config.env.evaluator_env_num = 2  # Reduce from 5 to 2 
        config.policy.collector_env_num = 2
        config.policy.evaluator_env_num = 2
        config.policy.n_episode = 2
        
        print(f"Experiment name: {config.exp_name}")
        print("Applied fixes:")
        print(f"- Multiprocessing: spawn method for macOS")
        print(f"- Reduced scale: {config.env.collector_env_num} collectors, {config.env.evaluator_env_num} evaluators")
        print(f"- Original env manager: {create_config.env_manager.type}")
        
        # Import and run training
        from lzero.entry import train_alphazero
        
        print("Starting training with simple fixes...")
        train_alphazero([config, create_config], seed=0, max_env_step=5000)  # Short test run
        
        print("✓ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("="*60)
    print("SIMPLE SPLENDOR ALPHAZERO TRAINING FIX")
    print("="*60)
    print("Applying minimal fixes:")
    print("1. Multiprocessing spawn method for macOS")
    print("2. Reduced environment count for testing")
    print("3. Keeping original subprocess environment manager")
    print("="*60)
    
    success = run_simple_training()
    
    print("="*60)
    if success:
        print("✓ SIMPLE FIX SUCCESSFUL")
        print("Training completed! Check logs for buffer statistics.")
    else:
        print("✗ SIMPLE FIX FAILED")
        print("May need more complex fixes.")
    print("="*60)
