#!/usr/bin/env python3
"""
Test training with timeout to prevent infinite waits
"""

import os
import sys
import time
import signal
import subprocess

def test_training_with_timeout(timeout_seconds=60):
    """Test training with a timeout"""
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Training test timed out after {timeout_seconds}s")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    start_time = time.time()
    
    try:
        # Run training script as subprocess
        cmd = [
            sys.executable, 'train_splendor_alphazero.py',
            '--max_env_step', '200',
            '--num_simulations', '5', 
            '--collector_env_num', '1',
            '--evaluator_env_num', '1',
            '--exp_name', 'timeout_test'
        ]
        
        print(f"🚀 Starting training test with {timeout_seconds}s timeout...")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
                # Look for success indicators
                if "buffer statistics is as follows:" in output and "pushed_in" in output:
                    # Check if we're getting data in buffer
                    next_line = process.stdout.readline()
                    if next_line and "0.000000" not in next_line:
                        print("✅ SUCCESS: Buffer is receiving data!")
                        process.terminate()
                        return True
                        
                if "Training completed successfully!" in output:
                    print("✅ SUCCESS: Training completed!")
                    return True
                    
                if "collected episode" in output.lower():
                    print("✅ SUCCESS: Episodes are being collected!")
                    process.terminate()
                    return True
        
        # Check return code
        return_code = process.poll()
        elapsed = time.time() - start_time
        
        if return_code == 0:
            print(f"✅ Training completed successfully in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ Training failed with return code {return_code} after {elapsed:.1f}s")
            return False
            
    except TimeoutError:
        print(f"❌ Training timed out after {timeout_seconds}s")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Training failed after {elapsed:.1f}s: {e}")
        return False
        
    finally:
        signal.alarm(0)  # Cancel timeout

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds')
    args = parser.parse_args()
    
    success = test_training_with_timeout(args.timeout)
    if success:
        print("\n🎉 TRAINING TEST PASSED!")
        print("The main training script can now run successfully.")
    else:
        print("\n❌ TRAINING TEST FAILED")
        print("Further investigation needed.")
    
    sys.exit(0 if success else 1)
