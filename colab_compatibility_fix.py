"""
🛠️ Colab Compatibility Fix for LightZero Dependencies

Use this if you get dependency conflicts.
"""

import subprocess
import sys

def install_with_no_deps(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package, "--no-deps"])

def force_install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package, "--force-reinstall", "--no-cache-dir"])

print("🛠️ Fixing common Colab compatibility issues...")

# Fix gymnasium version conflicts
force_install("gymnasium==0.28.0")

# Fix numpy/torch compatibility
force_install("numpy<2.0")
force_install("torch>=1.8.0")

# Install DI-engine with minimal dependencies
install_with_no_deps("DI-engine")

# Install specific LightZero dependencies
subprocess.run([sys.executable, "-m", "pip", "install", 
               "easydict", "pyyaml", "tensorboard", "wandb<0.20.0"])

# Clone LightZero with specific commit (more stable)
if not os.path.exists("LightZero"):
    subprocess.run(["git", "clone", "https://github.com/opendilab/LightZero.git"])
    
os.chdir("LightZero")
# Use specific stable commit
subprocess.run(["git", "checkout", "main"])  # or specific commit hash
install_with_no_deps(".")
os.chdir("..")

print("✅ Compatibility fixes applied!")

# Test imports
try:
    import lzero
    import ding
    print("✅ LightZero imports working!")
except Exception as e:
    print(f"❌ Still having issues: {e}")
    print("Try restarting runtime: Runtime -> Restart Runtime")
