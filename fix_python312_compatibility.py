#!/usr/bin/env python3
"""
Fix Python 3.12 compatibility issues for Splendor AlphaZero training

This script resolves lib2to3 and other Python 3.12 compatibility issues.
"""

import subprocess
import sys
import os

def fix_yapf_dependency():
    """Fix yapf dependency conflict"""
    print("🔧 Fixing yapf dependency for Python 3.12...")
    
    try:
        # Install compatible yapf version that doesn't use lib2to3
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'yapf>=0.40.0', '--force-reinstall', '--no-deps'
        ], check=True)
        print("✅ Updated yapf to Python 3.12 compatible version")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Could not update yapf, but training should still work")
        return False

def install_missing_dependencies():
    """Install any missing dependencies for Python 3.12"""
    print("📦 Installing Python 3.12 compatible dependencies...")
    
    # Dependencies that work well with Python 3.12
    compatible_deps = [
        'platformdirs>=3.5.1',
        'importlib-metadata>=4.0.0',  # For older packages
        'typing-extensions>=4.0.0',   # Type hints compatibility
    ]
    
    for dep in compatible_deps:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Could not install {dep}")

def create_lib2to3_stub():
    """Create a minimal lib2to3 stub for compatibility"""
    print("🔧 Creating lib2to3 compatibility stub...")
    
    try:
        import lib2to3
        print("✅ lib2to3 already available")
        return True
    except ImportError:
        pass
    
    # Create a minimal stub that prevents import errors
    stub_content = '''# Minimal lib2to3 stub for Python 3.12 compatibility
"""
This is a minimal stub for lib2to3 compatibility.
The actual lib2to3 functionality is not available in Python 3.12+.
"""

class ParseError(Exception):
    pass

def refactor_string(data, options=None):
    """Stub function that returns the input unchanged"""
    return data

# Other minimal stubs that some packages might expect
pytree = None
pygram = None
'''
    
    # Find site-packages directory
    import site
    site_packages = site.getsitepackages()[0]
    lib2to3_dir = os.path.join(site_packages, 'lib2to3')
    
    try:
        os.makedirs(lib2to3_dir, exist_ok=True)
        
        # Create __init__.py
        init_file = os.path.join(lib2to3_dir, '__init__.py')
        with open(init_file, 'w') as f:
            f.write(stub_content)
        
        # Create common modules that might be imported
        modules = ['refactor', 'pytree', 'pygram', 'patcomp']
        for module in modules:
            module_file = os.path.join(lib2to3_dir, f'{module}.py')
            with open(module_file, 'w') as f:
                f.write(f'# Stub for {module}\npass\n')
        
        print("✅ Created lib2to3 compatibility stub")
        return True
        
    except PermissionError:
        print("⚠️ No permission to create lib2to3 stub")
        print("   Training should still work, but you might see warnings")
        return False
    except Exception as e:
        print(f"⚠️ Could not create lib2to3 stub: {e}")
        return False

def test_imports():
    """Test critical imports to ensure everything works"""
    print("\n🔍 Testing critical imports...")
    
    tests = [
        ("LightZero", lambda: __import__('lzero')),
        ("DI-engine", lambda: __import__('ding')),
        ("Splendor Gym", lambda: __import__('splendor_gym')),
        ("PyTorch", lambda: __import__('torch')),
        ("NumPy", lambda: __import__('numpy')),
    ]
    
    all_passed = True
    for name, test_func in tests:
        try:
            # Add paths
            sys.path.insert(0, 'LightZero')
            sys.path.insert(0, '.')
            
            test_func()
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_passed = False
        except Exception as e:
            print(f"⚠️ {name}: {e}")
    
    return all_passed

def test_training_compatibility():
    """Test if training scripts can be imported without lib2to3 errors"""
    print("\n🔍 Testing training script compatibility...")
    
    try:
        # Test the config import that was failing
        sys.path.insert(0, 'LightZero')
        from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
        print("✅ Splendor config import")
        
        # Test environment creation
        from LightZero.zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
        env = SplendorLightZeroEnv()
        print("✅ Splendor environment creation")
        
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Fixing Python 3.12 Compatibility Issues")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    
    # Check if we're in the right directory
    if not os.path.exists('LightZero'):
        print("❌ LightZero directory not found!")
        print("Make sure you're in the splendor-gym directory")
        return False
    
    # Apply fixes
    success = True
    
    # Fix 1: Update yapf
    fix_yapf_dependency()
    
    # Fix 2: Install compatible dependencies
    install_missing_dependencies()
    
    # Fix 3: Create lib2to3 stub (fallback)
    create_lib2to3_stub()
    
    # Test the fixes
    if test_imports():
        print("\n✅ Basic imports working")
    else:
        print("\n⚠️ Some imports failed, but training might still work")
        success = False
    
    if test_training_compatibility():
        print("✅ Training compatibility verified")
    else:
        print("⚠️ Training compatibility issues detected")
        success = False
    
    if success:
        print("\n🎉 Python 3.12 compatibility fixes applied successfully!")
        print("\nYou can now run:")
        print("  python fix_ctree_imports.py")
        print("  python train_splendor_working.py")
    else:
        print("\n⚠️ Some compatibility issues remain")
        print("Training might still work, but you may see warnings")
        print("\nAlternative: Consider using Python 3.11 for full compatibility")
    
    return success

if __name__ == '__main__':
    main()
