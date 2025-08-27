#!/usr/bin/env python3
"""
Diagnostic script for Colab buffer issues

This script helps identify why buffer statistics show zeros in Google Colab
despite imports working correctly.
"""

import sys
import os
import time
import traceback
from contextlib import contextmanager

def setup_paths():
    """Setup import paths for Colab environment"""
    print("🔧 Setting up import paths...")
    
    # Add LightZero to path
    if 'LightZero' not in sys.path:
        sys.path.insert(0, 'LightZero')
    
    # Add current directory
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    print(f"Python paths: {sys.path[:3]}...")

@contextmanager
def timed_operation(name):
    """Context manager to time operations"""
    print(f"⏱️ Starting: {name}")
    start_time = time.time()
    try:
        yield
        elapsed = time.time() - start_time
        print(f"✅ Completed {name} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed {name} after {elapsed:.2f}s: {e}")
        raise

def test_basic_imports():
    """Test all critical imports"""
    print("\n📦 Testing imports...")
    
    imports_to_test = [
        ("LightZero base", lambda: __import__('lzero')),
        ("DI-engine", lambda: __import__('ding')),
        ("Splendor gym", lambda: __import__('splendor_gym')),
        ("NumPy", lambda: __import__('numpy')),
        ("PyTorch", lambda: __import__('torch')),
        ("Splendor config", lambda: __import__('zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config')),
        ("Splendor env", lambda: __import__('zoo.board_games.splendor.envs.splendor_lz_env')),
    ]
    
    for name, import_func in imports_to_test:
        try:
            with timed_operation(f"Import {name}"):
                import_func()
        except Exception as e:
            print(f"❌ Import failed for {name}: {e}")
            return False
    
    return True

def test_environment_creation():
    """Test basic environment creation"""
    print("\n🎮 Testing environment creation...")
    
    try:
        with timed_operation("Environment creation"):
            from zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
            env = SplendorLightZeroEnv()
            
            # Test reset
            obs = env.reset()
            print(f"✅ Environment reset successful, obs shape: {obs['observation'].shape}")
            
            # Test step
            action = 0  # First available action
            timestep = env.step(action)
            print(f"✅ Environment step successful, reward: {timestep.reward}")
            
            env.close()
            return True
            
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n⚙️ Testing config loading...")
    
    try:
        with timed_operation("Config loading"):
            from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config, splendor_alphazero_create_config
            
            print(f"✅ Config loaded successfully")
            print(f"   Environment manager type: {splendor_alphazero_config.env_manager.type}")
            print(f"   MCTS ctree: {splendor_alphazero_config.policy.mcts_ctree}")
            print(f"   Collector env num: {splendor_alphazero_config.collector_env_num}")
            
            return True
            
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return False

def test_minimal_collection():
    """Test minimal data collection"""
    print("\n📊 Testing minimal collection...")
    
    try:
        with timed_operation("Minimal collection setup"):
            # Import required components
            from lzero.entry import train_alphazero
            from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config, splendor_alphazero_create_config
            from ding.config import compile_config
            
            # Create minimal config for testing
            config = splendor_alphazero_config
            create_config = splendor_alphazero_create_config
            
            # Force minimal settings for testing
            config.collector_env_num = 1
            config.evaluator_env_num = 1
            config.policy.num_simulations = 2  # Minimal MCTS
            config.replay_buffer.replay_buffer_size = 100
            config.env.max_turns = 10  # Short games
            
            # Compile config
            cfg = compile_config(config, create_cfg=create_config, auto=True)
            
            print(f"✅ Config compiled for testing")
            print(f"   Collector envs: {cfg.collector_env_num}")
            print(f"   MCTS simulations: {cfg.policy.num_simulations}")
            print(f"   Max turns: {cfg.env.max_turns}")
            
            return cfg
            
    except Exception as e:
        print(f"❌ Minimal collection setup failed: {e}")
        traceback.print_exc()
        return None

def test_collector_creation(cfg):
    """Test collector creation"""
    print("\n🔄 Testing collector creation...")
    
    try:
        with timed_operation("Collector creation"):
            from lzero.worker import AlphaZeroCollector
            from ding.envs import SubprocessEnvManager, BaseEnvManager
            from ding.utils import set_pkg_seed
            
            # Set seed
            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
            
            # Create environment manager
            if cfg.env_manager.type == 'subprocess':
                from ding.envs import SubprocessEnvManager
                collector_env = SubprocessEnvManager(
                    env_fn=[lambda: create_env_fn(cfg) for _ in range(cfg.collector_env_num)],
                    cfg=cfg.env_manager
                )
            else:
                from ding.envs import BaseEnvManager
                collector_env = BaseEnvManager(
                    env_fn=[lambda: create_env_fn(cfg) for _ in range(cfg.collector_env_num)],
                    cfg=cfg.env_manager
                )
            
            print(f"✅ Environment manager created: {type(collector_env).__name__}")
            
            # Test environment manager
            collector_env.seed(cfg.seed)
            print(f"✅ Environment manager seeded")
            
            return collector_env
            
    except Exception as e:
        print(f"❌ Collector creation failed: {e}")
        traceback.print_exc()
        return None

def create_env_fn(cfg):
    """Create environment function"""
    from ding.envs import DingEnvWrapper
    from zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
    
    env = SplendorLightZeroEnv(cfg=cfg.env)
    env = DingEnvWrapper(env, cfg.env)
    return env

def test_single_episode_collection(collector_env):
    """Test collecting a single episode"""
    print("\n📈 Testing single episode collection...")
    
    try:
        with timed_operation("Single episode collection"):
            # Reset environments
            obs = collector_env.reset()
            print(f"✅ Environments reset, obs shape: {[o['observation'].shape for o in obs]}")
            
            # Run a few steps
            steps = 0
            max_steps = 20
            transitions = []
            
            while steps < max_steps:
                # Simple random actions for testing
                actions = [0] * len(obs)  # All environments take action 0
                
                timesteps = collector_env.step(actions)
                transitions.extend(timesteps)
                
                steps += 1
                
                # Check if any episode is done
                if any(ts.done for ts in timesteps):
                    print(f"✅ Episode completed after {steps} steps")
                    break
                
                # Get new observations for next step
                obs = [ts.obs for ts in timesteps if not ts.done]
                if not obs:
                    break
            
            print(f"✅ Collected {len(transitions)} transitions")
            
            # Check transition structure
            if transitions:
                first_transition = transitions[0]
                print(f"   First transition keys: {list(first_transition.info.keys()) if hasattr(first_transition, 'info') else 'No info'}")
                print(f"   Observation shape: {first_transition.obs['observation'].shape}")
                print(f"   Reward: {first_transition.reward}")
                print(f"   Done: {first_transition.done}")
            
            collector_env.close()
            return len(transitions) > 0
            
    except Exception as e:
        print(f"❌ Single episode collection failed: {e}")
        traceback.print_exc()
        if 'collector_env' in locals():
            try:
                collector_env.close()
            except:
                pass
        return False

def check_environment_info():
    """Check environment information for debugging"""
    print("\n🔍 Checking environment info...")
    
    try:
        import platform
        import multiprocessing as mp
        
        print(f"Platform: {platform.platform()}")
        print(f"Python version: {sys.version}")
        print(f"Multiprocessing start method: {mp.get_start_method()}")
        print(f"CPU count: {mp.cpu_count()}")
        
        # Check if we're in Colab
        try:
            import google.colab
            print("✅ Running in Google Colab")
            colab_env = True
        except ImportError:
            print("ℹ️ Not running in Google Colab")
            colab_env = False
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"Available memory: {memory.available / (1024**3):.1f} GB")
        except ImportError:
            print("psutil not available for memory check")
        
        return colab_env
        
    except Exception as e:
        print(f"⚠️ Environment check failed: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("🩺 COLAB BUFFER DIAGNOSTIC")
    print("=" * 50)
    
    # Setup
    setup_paths()
    
    # Check environment
    is_colab = check_environment_info()
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ DIAGNOSIS: Import failures detected")
        return False
    
    # Test environment creation
    if not test_environment_creation():
        print("\n❌ DIAGNOSIS: Environment creation issues")
        return False
    
    # Test config loading
    if not test_config_loading():
        print("\n❌ DIAGNOSIS: Config loading issues")
        return False
    
    # Test minimal collection setup
    cfg = test_minimal_collection()
    if cfg is None:
        print("\n❌ DIAGNOSIS: Collection setup issues")
        return False
    
    # Test collector creation
    collector_env = test_collector_creation(cfg)
    if collector_env is None:
        print("\n❌ DIAGNOSIS: Collector creation issues")
        return False
    
    # Test single episode collection
    if not test_single_episode_collection(collector_env):
        print("\n❌ DIAGNOSIS: Episode collection issues")
        return False
    
    print("\n✅ DIAGNOSIS COMPLETE: All tests passed!")
    print("\nIf buffer still shows zeros, the issue might be:")
    print("1. Training loop configuration")
    print("2. Buffer initialization timing")
    print("3. Data serialization in the training pipeline")
    print("4. Colab-specific multiprocessing limitations")
    
    return True

if __name__ == '__main__':
    main()
