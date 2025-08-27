#!/usr/bin/env python3
"""
Minimal training test specifically for Colab environment

This script runs a very minimal training session to isolate buffer issues.
"""

import sys
import os
import time
import traceback

def setup_colab_environment():
    """Setup environment for Colab"""
    print("🔧 Setting up Colab environment...")
    
    # Add paths
    sys.path.insert(0, 'LightZero')
    sys.path.insert(0, '.')
    
    # Apply fixes if needed
    try:
        # Check if fixes are needed
        from lzero.mcts.ctree.ctree_efficientzero import ez_tree
        print("✅ ctree imports working")
    except ImportError:
        print("🔧 Applying ctree import fix...")
        os.system('python fix_ctree_imports.py')
    
    try:
        import lib2to3
        print("✅ lib2to3 available")
    except ImportError:
        print("🔧 Applying Python 3.12 compatibility fix...")
        os.system('python fix_python312_compatibility.py')

def run_minimal_training():
    """Run minimal training session"""
    print("\n🚀 Running minimal training...")
    
    try:
        # Import training components
        from lzero.entry import train_alphazero
        from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
            splendor_alphazero_config, 
            splendor_alphazero_create_config
        )
        
        # Create ultra-minimal config for testing
        config = splendor_alphazero_config.copy()
        create_config = splendor_alphazero_create_config.copy()
        
        # Minimal settings for quick testing
        config.max_env_step = 50  # Very small
        config.collector_env_num = 1
        config.evaluator_env_num = 1
        config.policy.num_simulations = 3  # Minimal MCTS
        config.replay_buffer.replay_buffer_size = 50
        config.batch_size = 8
        config.env.max_turns = 15  # Short games
        config.env_manager.type = 'base'  # Avoid subprocess issues
        
        # Ensure Python MCTS
        config.policy.mcts_ctree = False
        
        # Set experiment name
        config.exp_name = 'colab_minimal_test'
        
        print(f"⚙️ Config settings:")
        print(f"   Max env steps: {config.max_env_step}")
        print(f"   Collector envs: {config.collector_env_num}")
        print(f"   MCTS simulations: {config.policy.num_simulations}")
        print(f"   Max turns: {config.env.max_turns}")
        print(f"   Environment manager: {config.env_manager.type}")
        print(f"   MCTS ctree: {config.policy.mcts_ctree}")
        
        # Start training with timeout
        print("\n🎯 Starting minimal training (should see buffer activity quickly)...")
        
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        
        try:
            train_alphazero([config, create_config], seed=0, max_train_iter=1)
            elapsed = time.time() - start_time
            print(f"\n✅ Training completed successfully in {elapsed:.1f}s")
            return True
            
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\n⚠️ Training interrupted after {elapsed:.1f}s")
            return False
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Training failed after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        traceback.print_exc()
        return False

def monitor_buffer_logs():
    """Monitor for buffer logs in the output"""
    print("\n📊 Buffer monitoring tips:")
    print("1. Look for 'buffer statistics' messages")
    print("2. 'pushed_in' should be > 0 within first few minutes")
    print("3. If 'pushed_in' stays 0.0, collection is not working")
    print("4. Check for any error messages during collection")

def main():
    """Main function"""
    print("🧪 COLAB MINIMAL TRAINING TEST")
    print("=" * 50)
    
    # Setup
    setup_colab_environment()
    
    # Monitor instructions
    monitor_buffer_logs()
    
    # Run test
    if run_minimal_training():
        print("\n🎉 SUCCESS: Training ran without major errors")
        print("Check the buffer statistics in the output above.")
        print("If buffer shows zeros, the issue is in data collection.")
    else:
        print("\n❌ FAILURE: Training encountered errors")
        print("Check the error messages above for clues.")
    
    print("\n🔍 Next steps if buffer still shows zeros:")
    print("1. Run 'python diagnose_colab_buffer.py' for detailed diagnostics")
    print("2. Check if environment episodes are completing")
    print("3. Verify data serialization is working")
    print("4. Consider memory or multiprocessing constraints in Colab")

if __name__ == '__main__':
    main()
