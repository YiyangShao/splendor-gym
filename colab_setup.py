#!/usr/bin/env python3
"""
Colab Setup Script for Splendor AlphaZero

Run this in the first cell of your Colab notebook:
!python colab_setup.py
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"Error: {result.stderr}")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def main():
    print("🚀 Setting up Splendor AlphaZero for Google Colab...")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️  Not running in Colab, but continuing anyway...")
    
    # Install core dependencies
    print("\n📦 Installing core dependencies...")
    run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    run_command("pip install gymnasium==0.28.0 easydict pyyaml numpy tensorboard")
    
    # Install DI-engine
    print("\n🔧 Installing DI-engine...")
    run_command("pip install DI-engine")
    
    # Clone LightZero if not exists
    if not os.path.exists("LightZero"):
        print("\n📥 Cloning LightZero...")
        run_command("git clone https://github.com/opendilab/LightZero.git")
    
    # Install LightZero
    print("\n⚙️ Installing LightZero...")
    os.chdir("LightZero")
    run_command("pip install -e . --no-deps")
    
    # Install additional dependencies
    run_command("pip install pympler xxhash minigrid line_profiler")
    os.chdir("..")
    
    # Clone splendor-gym if not exists
    if not os.path.exists("splendor-gym"):
        print("\n🎮 Setting up Splendor environment...")
        # You'll need to replace this with your actual repo URL
        repo_url = input("Enter your splendor-gym repository URL (or press Enter to skip): ").strip()
        if repo_url:
            run_command(f"git clone {repo_url} splendor-gym")
        else:
            print("⚠️  Skipping splendor-gym clone. You'll need to upload your code manually.")
            return
    
    os.chdir("splendor-gym")
    run_command("pip install -e .")
    
    print("\n✅ Setup complete! You can now run:")
    print("python train_splendor_alphazero.py --max_env_step 10000")
    
    # Test import
    print("\n🧪 Testing imports...")
    try:
        import lzero
        print("✅ LightZero imported successfully")
        
        from LightZero.zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
        print("✅ Splendor environment imported successfully")
        
        print("\n🎉 All systems ready for training!")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print("You may need to restart the runtime and try again.")

if __name__ == "__main__":
    main()
