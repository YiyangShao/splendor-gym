#!/usr/bin/env python3
"""
Debug exactly where collection is hanging
"""

import sys
import os
import time
import signal
sys.path.insert(0, 'LightZero')

def debug_collection_step_by_step():
    """Debug collection process step by step with detailed timing"""
    
    print("=== DEBUG: Collection Hang Analysis ===")
    
    # Import and setup
    from LightZero.zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.policy import create_policy
    from ding.utils import set_pkg_seed
    from lzero.worker import AlphaZeroCollector
    from lzero.policy import visit_count_temperature
    from functools import partial
    from easydict import EasyDict
    import copy
    
    # Minimal config
    cfg = EasyDict(copy.deepcopy(splendor_alphazero_config))
    create_cfg = EasyDict(copy.deepcopy(splendor_alphazero_create_config))
    
    cfg.env.collector_env_num = 1
    cfg.policy.collector_env_num = 1
    cfg.policy.mcts.num_simulations = 5
    cfg.policy.n_episode = 1
    cfg.exp_name = f'debug_hang_{int(time.time())}'
    cfg.seed = 42
    cfg.policy.device = 'cpu'
    cfg.policy.cuda = False
    
    print("✓ Config created")
    
    # Compile config
    start = time.time()
    cfg = compile_config(cfg, seed=cfg.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
    print(f"✓ Config compiled in {time.time() - start:.2f}s")
    
    # Create environment manager
    start = time.time()
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(cfg.seed)
    print(f"✓ Environment manager created in {time.time() - start:.2f}s")
    
    # Create policy
    start = time.time()
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval'])
    print(f"✓ Policy created in {time.time() - start:.2f}s")
    
    # Create collector
    start = time.time()
    collector = AlphaZeroCollector(
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=None,
        exp_name=cfg.exp_name,
    )
    print(f"✓ Collector created in {time.time() - start:.2f}s")
    
    # Test environment readiness
    start = time.time()
    obs = collector_env.ready_obs
    print(f"✓ Environment ready obs obtained in {time.time() - start:.2f}s")
    print(f"  - Ready obs keys: {list(obs.keys())}")
    print(f"  - Obs shapes: {[str(type(v)) for v in obs.values()]}")
    
    # Test policy forward (this is often where it hangs)
    print("\n--- Testing Policy Forward ---")
    start = time.time()
    
    try:
        temperature = visit_count_temperature(
            cfg.policy.manual_temperature_decay,
            cfg.policy.fixed_temperature_value, 
            cfg.policy.threshold_training_steps_for_final_temperature,
            trained_steps=0
        )
        print(f"✓ Temperature calculated: {temperature}")
        
        # This is often where it hangs
        print("🔥 About to call policy forward - this often hangs...")
        policy_output = policy.collect_mode.forward(obs, temperature)
        forward_time = time.time() - start
        print(f"✅ Policy forward completed in {forward_time:.2f}s!")
        print(f"  - Policy output keys: {list(policy_output.keys())}")
        
        # Test environment step
        print("\n--- Testing Environment Step ---")
        start = time.time()
        actions = {env_id: output['action'] for env_id, output in policy_output.items()}
        print(f"✓ Actions extracted: {actions}")
        
        timesteps = collector_env.step(actions)
        step_time = time.time() - start
        print(f"✅ Environment step completed in {step_time:.2f}s!")
        
        # Check timestep
        for env_id, ts in timesteps.items():
            print(f"  Env {env_id}: done={ts.done}, reward={ts.reward}, info_keys={list(ts.info.keys())}")
        
        # Now try the actual collection
        print("\n--- Testing Full Collection ---")
        collect_kwargs = {'temperature': temperature}
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Collection timed out!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        start = time.time()
        try:
            print("🔥 Starting collection - this often hangs...")
            new_data = collector.collect(n_episode=1, train_iter=0, policy_kwargs=collect_kwargs)
            collection_time = time.time() - start
            signal.alarm(0)
            
            print(f"✅ Collection completed in {collection_time:.2f}s!")
            if new_data and len(new_data) > 0:
                total_transitions = sum(len(episode) for episode in new_data)
                print(f"  - Episodes collected: {len(new_data)}")
                print(f"  - Total transitions: {total_transitions}")
                return True
            else:
                print("❌ Collection returned no data")
                return False
                
        except TimeoutError:
            signal.alarm(0)
            print("❌ Collection timed out after 30s")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        collector_env.close()

if __name__ == '__main__':
    try:
        success = debug_collection_step_by_step()
        if success:
            print("\n🎉 SUCCESS: Collection works!")
        else:
            print("\n❌ FAILED: Collection still has issues")
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n💥 FAILED: {e}")
        import traceback
        traceback.print_exc()
