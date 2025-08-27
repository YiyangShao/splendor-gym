#!/usr/bin/env python3
"""
Google Colab Setup Script for Splendor AlphaZero

This script sets up the environment for running Splendor AlphaZero training in Google Colab.
It handles all the necessary dependencies and configuration automatically.

Usage in Colab:
1. !git clone <your-repo-url>
2. %cd splendor-gym
3. !python colab_setup.py
4. !python train_splendor_colab.py
"""

import os
import sys
import subprocess
import importlib.util

def install_dependencies():
    """Install required dependencies for Colab"""
    print("📦 Installing dependencies for Google Colab...")
    
    # Core dependencies
    dependencies = [
        'torch>=1.9.0',
        'numpy>=1.21.0', 
        'easydict',
        'tensorboard',
        'gym',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pyyaml',
        'cloudpickle',
        'h5py',
        'scipy',
        'scikit-learn',
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            print(f"✅ {dep}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {dep}: {e}")
    
    # Install DI-engine (LightZero dependency)
    try:
        print("Installing DI-engine...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'DI-engine'], 
                      check=True, capture_output=True)
        print("✅ DI-engine")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to install DI-engine: {e}")

def setup_lightzero():
    """Setup LightZero without C++ compilation"""
    print("🔧 Setting up LightZero for Colab...")
    
    if not os.path.exists('LightZero'):
        print("❌ LightZero directory not found!")
        print("Make sure you're in the correct directory")
        return False
    
    # Add to Python path
    sys.path.insert(0, 'LightZero')
    
    # Test basic import
    try:
        import lzero
        print("✅ LightZero imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LightZero: {e}")
        return False

def setup_splendor_gym():
    """Setup splendor_gym"""
    print("🎮 Setting up Splendor environment...")
    
    if not os.path.exists('splendor_gym'):
        print("❌ splendor_gym directory not found!")
        return False
    
    # Install in development mode
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                      check=True, capture_output=True)
        print("✅ splendor_gym installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to install splendor_gym: {e}")
        # Try to add to path manually
        sys.path.insert(0, '.')
        try:
            import splendor_gym
            print("✅ splendor_gym imported via path")
            return True
        except ImportError:
            print("❌ Failed to import splendor_gym")
            return False

def patch_ctree_imports():
    """Patch ctree imports to be conditional"""
    print("🔧 Patching ctree imports for Colab compatibility...")
    
    mcts_ctree_file = 'LightZero/lzero/mcts/tree_search/mcts_ctree.py'
    
    if not os.path.exists(mcts_ctree_file):
        print("⚠️ mcts_ctree.py not found, skipping patch")
        return True
    
    with open(mcts_ctree_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'CTREE_EFFICIENTZERO_AVAILABLE' in content:
        print("✅ mcts_ctree.py already patched")
        return True
    
    # Apply patch
    old_line = "from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero"
    
    if old_line in content:
        patch = '''# Conditional imports for C++ extensions (Colab-compatible)
try:
    from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
    CTREE_EFFICIENTZERO_AVAILABLE = True
except ImportError:
    tree_efficientzero = None
    CTREE_EFFICIENTZERO_AVAILABLE = False
    print("ℹ️ Using Python MCTS (ctree not available)")'''
        
        new_content = content.replace(old_line, patch)
        
        # Similar patches for other ctree imports
        old_line2 = "from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero"
        if old_line2 in new_content:
            patch2 = '''try:
    from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero
    CTREE_GUMBEL_MUZERO_AVAILABLE = True
except ImportError:
    tree_gumbel_muzero = None
    CTREE_GUMBEL_MUZERO_AVAILABLE = False'''
            new_content = new_content.replace(old_line2, patch2)
        
        old_line3 = "from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero"
        if old_line3 in new_content:
            patch3 = '''try:
    from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero
    CTREE_MUZERO_AVAILABLE = True
except ImportError:
    tree_muzero = None
    CTREE_MUZERO_AVAILABLE = False'''
            new_content = new_content.replace(old_line3, patch3)
        
        # Backup and write
        with open(mcts_ctree_file + '.backup', 'w') as f:
            f.write(content)
        
        with open(mcts_ctree_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Applied ctree compatibility patch")
        return True
    else:
        print("⚠️ mcts_ctree.py structure has changed, manual patch needed")
        return False

def verify_setup():
    """Verify that everything is set up correctly"""
    print("🔍 Verifying setup...")
    
    tests = [
        ("LightZero", lambda: __import__('lzero')),
        ("Splendor Gym", lambda: __import__('splendor_gym')),
        ("DI-engine", lambda: __import__('ding')),
        ("PyTorch", lambda: __import__('torch')),
        ("NumPy", lambda: __import__('numpy')),
        ("EasyDict", lambda: __import__('easydict')),
    ]
    
    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_passed = False
    
    # Test Splendor environment creation
    try:
        from LightZero.zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
        env = SplendorLightZeroEnv()
        env.reset()
        print("✅ Splendor environment creation")
    except Exception as e:
        print(f"❌ Splendor environment: {e}")
        all_passed = False
    
    # Test config loading
    try:
        from LightZero.zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
        print("✅ Splendor config loading")
    except Exception as e:
        print(f"❌ Splendor config: {e}")
        all_passed = False
    
    return all_passed

def main():
    """Main setup function"""
    print("🚀 Setting up Splendor AlphaZero for Google Colab")
    print("=" * 60)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("ℹ️ Not in Google Colab (local setup)")
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Setup LightZero
    if not setup_lightzero():
        print("❌ LightZero setup failed")
        return False
    
    # Step 3: Setup Splendor Gym
    if not setup_splendor_gym():
        print("❌ Splendor Gym setup failed")
        return False
    
    # Step 4: Patch ctree imports
    patch_ctree_imports()
    
    # Step 5: Verify setup
    if verify_setup():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run training: !python train_splendor_colab.py")
        print("2. Monitor with: !tensorboard --logdir data_az_ctree/")
        print("\nFor custom settings:")
        print("!python train_splendor_colab.py --num_simulations 25 --max_turns 60")
        return True
        else:
        print("\n❌ Setup verification failed")
        print("Some components may not work correctly")
        return False

if __name__ == '__main__':
    main()