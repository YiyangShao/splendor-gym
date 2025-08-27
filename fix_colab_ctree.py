#!/usr/bin/env python3
"""
Fix LightZero ctree import issues in Google Colab

This script handles the common ctree compilation issues when running
LightZero in different environments (especially Google Colab).
"""

import sys
import os
import subprocess
import importlib.util

def check_ctree_availability():
    """Check which ctree modules are available"""
    print("🔍 Checking ctree module availability...")
    
    ctree_modules = [
        'lzero.mcts.ctree.ctree_efficientzero.ez_tree',
        'lzero.mcts.ctree.ctree_muzero.mz_tree', 
        'lzero.mcts.ctree.ctree_gumbel_muzero.gmz_tree',
    ]
    
    available = []
    missing = []
    
    for module in ctree_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                # Try to actually import it
                exec(f"import {module}")
                available.append(module)
                print(f"✅ {module}")
            else:
                missing.append(module)
                print(f"❌ {module} - spec not found")
        except ImportError as e:
            missing.append(module)
            print(f"❌ {module} - import error: {e}")
        except Exception as e:
            missing.append(module)
            print(f"❌ {module} - other error: {e}")
    
    return available, missing

def install_build_dependencies():
    """Install dependencies needed for building ctree"""
    print("\n📦 Installing build dependencies...")
    
    try:
        # Install build essentials
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'Cython', 'numpy', 'setuptools', 'wheel'
        ], check=True)
        print("✅ Build dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def compile_ctree_modules():
    """Attempt to compile ctree modules"""
    print("\n🔨 Attempting to compile ctree modules...")
    
    # Check if we're in the right directory
    if not os.path.exists('LightZero'):
        print("❌ LightZero directory not found. Please run from the project root.")
        return False
    
    try:
        # Try to build the extensions
        os.chdir('LightZero')
        
        # Method 1: Try setup.py build_ext
        try:
            result = subprocess.run([
                sys.executable, 'setup.py', 'build_ext', '--inplace'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Compiled ctree modules successfully with setup.py")
                return True
            else:
                print(f"⚠️ setup.py failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️ setup.py method failed: {e}")
        
        # Method 2: Try pip install in editable mode
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-e', '.'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Compiled ctree modules successfully with pip install -e")
                return True
            else:
                print(f"⚠️ pip install -e failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️ pip install method failed: {e}")
            
        return False
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False
    finally:
        os.chdir('..')

def use_python_fallback():
    """Configure LightZero to use Python-only MCTS (slower but works)"""
    print("\n🐍 Setting up Python-only MCTS fallback...")
    
    try:
        # Check if we can import the base MCTS
        from lzero.mcts.ptree import ptree_alphazero
        print("✅ Python tree (ptree) available as fallback")
        
        # Create a simple config to use ptree instead of ctree
        fallback_config = """
# Add this to your config to use Python trees instead of C++ trees
config.policy.mcts_ctree = False  # Use Python trees
"""
        
        with open('use_python_trees_config.txt', 'w') as f:
            f.write(fallback_config)
            
        print("✅ Fallback configuration saved to 'use_python_trees_config.txt'")
        print("💡 Modify your training config to set: config.policy.mcts_ctree = False")
        return True
        
    except ImportError:
        print("❌ Python tree fallback also not available")
        return False

def fix_import_path():
    """Fix potential import path issues"""
    print("\n🔧 Fixing import paths...")
    
    # Add current directory to Python path
    current_dir = os.getcwd()
    lightzero_path = os.path.join(current_dir, 'LightZero')
    
    if lightzero_path not in sys.path:
        sys.path.insert(0, lightzero_path)
        print(f"✅ Added {lightzero_path} to Python path")
    
    # Check if we can import LightZero base
    try:
        import lzero
        print(f"✅ LightZero base imported from: {lzero.__file__}")
        return True
    except ImportError as e:
        print(f"❌ Cannot import LightZero base: {e}")
        return False

def main():
    """Main function to fix ctree issues"""
    print("🔧 LightZero ctree Fix Tool")
    print("=" * 50)
    
    # Step 1: Fix import paths
    if not fix_import_path():
        print("\n❌ Cannot fix import paths. Please check your setup.")
        return
    
    # Step 2: Check current status
    available, missing = check_ctree_availability()
    
    if not missing:
        print("\n🎉 All ctree modules are available! No fixes needed.")
        return
    
    print(f"\n⚠️ Missing ctree modules: {len(missing)}")
    print("Attempting fixes...")
    
    # Step 3: Install dependencies
    if not install_build_dependencies():
        print("❌ Failed to install build dependencies")
        use_python_fallback()
        return
    
    # Step 4: Try to compile
    if compile_ctree_modules():
        print("\n🎉 ctree compilation successful!")
        
        # Verify the fix worked
        available_after, missing_after = check_ctree_availability()
        if not missing_after:
            print("✅ All ctree modules now working!")
        else:
            print(f"⚠️ Still missing: {missing_after}")
            use_python_fallback()
    else:
        print("\n❌ ctree compilation failed")
        use_python_fallback()
    
    print("\n📋 Summary:")
    print("If ctree compilation worked: Use normal training")
    print("If using Python fallback: Add 'config.policy.mcts_ctree = False' to your config")
    print("Python trees are slower but functionally equivalent")

if __name__ == '__main__':
    main()
