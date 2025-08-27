"""
🎮 Splendor AlphaZero Training for Google Colab

Copy this entire script into a Colab cell and run it.
This handles installation and training in one go.
"""

# 🚀 STEP 1: Install everything
print("🚀 Installing dependencies...")

import subprocess
import sys
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Core dependencies
install_package("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
install_package("gymnasium==0.28.0")
install_package("easydict")
install_package("pyyaml")
install_package("numpy")
install_package("tensorboard")
install_package("DI-engine")

# Clone LightZero
if not os.path.exists("LightZero"):
    subprocess.run(["git", "clone", "https://github.com/opendilab/LightZero.git"])

# Install LightZero
os.chdir("LightZero")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
install_package("pympler xxhash minigrid line_profiler")
os.chdir("..")

print("✅ Installation complete!")

# 🎮 STEP 2: Setup Splendor Environment
print("🎮 Setting up Splendor environment...")

# You'll need to upload your splendor-gym files to Colab
# Or clone from your GitHub repo:
# subprocess.run(["git", "clone", "YOUR_GITHUB_REPO_URL", "splendor-gym"])

# For now, create a minimal setup
if not os.path.exists("splendor-gym"):
    print("⚠️ Please upload your splendor-gym folder to Colab first!")
    print("Or modify this script to clone from your GitHub repo.")
    sys.exit(1)

os.chdir("splendor-gym")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])

# 🧪 STEP 3: Test everything
print("🧪 Testing imports...")
try:
    import lzero
    from LightZero.zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
    from LightZero.zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
    
    print("✅ All imports successful!")
    
    # Test environment
    env = SplendorLightZeroEnv(splendor_alphazero_config.env)
    obs = env.reset()
    print(f"✅ Environment test passed! Obs shape: {obs['observation'].shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("You may need to restart runtime and try again.")
    sys.exit(1)

# 🏋️ STEP 4: Start Training
print("🏋️ Starting training...")

# Colab-optimized parameters (smaller for free tier)
training_command = [
    sys.executable, "train_splendor_alphazero.py",
    "--max_env_step", "20000",        # Smaller for Colab
    "--num_simulations", "50",        # Reduced for speed
    "--collector_env_num", "2",       # Fewer processes
    "--evaluator_env_num", "1",       # Minimal evaluation
    "--batch_size", "64",             # Smaller batch
    "--exp_name", "colab_experiment"
]

subprocess.run(training_command)

print("🎉 Training complete! Check the data_az_ctree/ directory for results.")

# 📊 STEP 5: Show results
print("📊 Training results:")
if os.path.exists("data_az_ctree"):
    subprocess.run(["ls", "-la", "data_az_ctree/"])
    print("\n🔍 To view training logs:")
    print("from tensorboard import notebook")
    print("notebook.display(port=6006, height=1000)")
else:
    print("No training results found.")
