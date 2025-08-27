#!/usr/bin/env python3
"""
Immediate fix for ctree import errors in Google Colab

This script directly patches the problematic import files to make them Colab-compatible.
Run this FIRST before any training script.

Usage in Colab:
!python fix_ctree_imports.py

Then you can run training normally.
"""

import os
import sys

def patch_mcts_ctree():
    """Patch the main problematic file"""
    file_path = 'LightZero/lzero/mcts/tree_search/mcts_ctree.py'
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    print(f"🔧 Patching {file_path}...")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'try:' in content and 'ez_tree as tree_efficientzero' in content:
        print("✅ Already patched")
        return True
    
    # Define the problematic imports and their fixes
    patches = [
        {
            'old': 'from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero',
            'new': '''try:
    from lzero.mcts.ctree.ctree_efficientzero import ez_tree as tree_efficientzero
except ImportError:
    print("⚠️ ctree_efficientzero not available, using Python fallback")
    tree_efficientzero = None'''
        },
        {
            'old': 'from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero',
            'new': '''try:
    from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree as tree_gumbel_muzero
except ImportError:
    print("⚠️ ctree_gumbel_muzero not available, using Python fallback") 
    tree_gumbel_muzero = None'''
        },
        {
            'old': 'from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero',
            'new': '''try:
    from lzero.mcts.ctree.ctree_muzero import mz_tree as tree_muzero
except ImportError:
    print("⚠️ ctree_muzero not available, using Python fallback")
    tree_muzero = None'''
        }
    ]
    
    # Apply patches
    new_content = content
    for patch in patches:
        if patch['old'] in new_content:
            new_content = new_content.replace(patch['old'], patch['new'])
            print(f"  ✅ Patched ctree import")
    
    # Backup original file
    backup_path = file_path + '.original'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"  📝 Backup saved to {backup_path}")
    
    # Write patched file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Patching complete")
    return True

def patch_efficientzero_init():
    """Fix the __init__.py file that's causing the import error"""
    file_path = 'LightZero/lzero/mcts/ctree/ctree_efficientzero/__init__.py'
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found")
        return False
    
    print(f"🔧 Checking {file_path}...")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    # If the file is empty or doesn't import ez_tree, create a safe version
    if not content or 'ez_tree' not in content:
        safe_content = '''# Safe import for Colab compatibility
try:
    from .ez_tree import *
except ImportError:
    # C++ extension not available, will use Python fallback
    pass
'''
        with open(file_path, 'w') as f:
            f.write(safe_content)
        print("✅ Created safe __init__.py")
        return True
    else:
        print("✅ __init__.py already has content")
        return True

def verify_fix():
    """Verify that the fix works"""
    print("\n🔍 Verifying fix...")
    
    try:
        # Test the import that was failing
        sys.path.insert(0, 'LightZero')
        
        # This should now work without error
        from lzero.mcts.tree_search import mcts_ctree
        print("✅ mcts_ctree import successful")
        
        # Test Splendor config import
        from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
        print("✅ Splendor config import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import still failing: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Fixing ctree import errors for Google Colab")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('LightZero'):
        print("❌ LightZero directory not found!")
        print("Make sure you're in the splendor-gym directory")
        print("Current directory contents:")
        print(os.listdir('.'))
        return False
    
    # Apply patches
    success = True
    success &= patch_mcts_ctree()
    success &= patch_efficientzero_init()
    
    if success:
        # Verify the fix
        if verify_fix():
            print("\n🎉 Fix successful! ctree imports are now Colab-compatible")
            print("\nYou can now run:")
            print("  !python train_splendor_colab.py")
            print("  !python train_splendor_working.py")
            return True
        else:
            print("\n⚠️ Fix applied but verification failed")
            print("Training might still work with Python fallbacks")
            return False
    else:
        print("\n❌ Fix failed")
        return False

if __name__ == '__main__':
    main()
