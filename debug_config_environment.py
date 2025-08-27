#!/usr/bin/env python3
"""
Test if environment through config system behaves differently than direct creation
"""

import sys
sys.path.insert(0, 'LightZero')

def test_environment_through_config():
    """Test environment created through the config system like in real training"""
    
    print("=== Testing Environment Through Config System ===")
    
    from LightZero.zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting, create_env_manager
    from functools import partial
    from easydict import EasyDict
    import copy
    
    # Use same config as training
    cfg = EasyDict(copy.deepcopy(splendor_alphazero_config))
    create_cfg = EasyDict(copy.deepcopy(splendor_alphazero_create_config))
    
    cfg.env.collector_env_num = 1
    cfg.exp_name = 'config_test'
    cfg.seed = 42
    
    print(f"Config max_turns: {cfg.env.get('max_turns', 'NOT_SET')}")
    
    # Compile config
    cfg = compile_config(cfg, seed=cfg.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
    
    print(f"Compiled config max_turns: {cfg.env.get('max_turns', 'NOT_SET')}")
    
    # Create environment manager like in training
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(cfg.seed)
    
    print(f"Env manager type: {type(collector_env)}")
    print(f"Number of environments: {collector_env._env_num}")
    
    # Test the actual environment instance  
    print(f"Environment type: {type(collector_env._env_ref)}")
    print(f"Environment max_turns: {getattr(collector_env._env_ref, 'max_turns', 'NOT_SET')}")
    
    # Test environment reset and step
    obs = collector_env.ready_obs
    print(f"Ready obs keys: {list(obs.keys())}")
    
    # Test manual episode on the actual environment
    print("\n--- Testing Episode on Config-Created Environment ---")
    
    env_ref = collector_env._env_ref  # Get the environment reference
    
    # Reset the specific environment
    collector_env.reset({0: None})
    obs = collector_env.ready_obs[0]
    
    print(f"Environment max_turns: {getattr(env_ref, 'max_turns', 'NOT_SET')}")
    print(f"Initial turn_count: {getattr(env_ref.state, 'turn_count', 'NOT_SET')}")
    
    step_count = 0
    max_steps = 10  # Test just a few steps
    
    for step in range(max_steps):
        # Get legal actions from environment
        legal_actions = getattr(env_ref, 'legal_actions', [])
        if not legal_actions:
            print(f"No legal actions at step {step}")
            break
            
        action = legal_actions[0]
        print(f"Step {step + 1}: turn_count={getattr(env_ref.state, 'turn_count', 'N/A')}, action={action}")
        
        # Step through environment manager
        timesteps = collector_env.step({0: action})
        timestep = timesteps[0]
        
        print(f"  Result: done={timestep.done}, reward={timestep.reward}")
        print(f"  New turn_count: {getattr(env_ref.state, 'turn_count', 'N/A')}")
        print(f"  Game over: {getattr(env_ref.state, 'game_over', 'N/A')}")
        
        if timestep.done:
            print(f"✅ Episode completed after {step + 1} steps!")
            print(f"  Turn limit reached: {getattr(env_ref.state, 'turn_limit_reached', 'N/A')}")
            print(f"  eval_episode_return: {timestep.info.get('eval_episode_return', 'NOT_SET')}")
            collector_env.close()
            return True
            
        step_count += 1
    
    print(f"Episode didn't complete in {max_steps} steps")
    collector_env.close()
    return False

def test_forced_short_episode():
    """Test with forced very short max_turns"""
    
    print("\n=== Testing with Forced Short max_turns ===")
    
    from LightZero.zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting, create_env_manager
    from functools import partial
    from easydict import EasyDict
    import copy
    
    # Force short episodes
    cfg = EasyDict(copy.deepcopy(splendor_alphazero_config))
    create_cfg = EasyDict(copy.deepcopy(splendor_alphazero_create_config))
    
    cfg.env.collector_env_num = 1
    cfg.env.max_turns = 3  # Force very short episodes
    cfg.exp_name = 'short_test'
    cfg.seed = 42
    
    print(f"Forcing max_turns to: {cfg.env.max_turns}")
    
    # Compile config
    cfg = compile_config(cfg, seed=cfg.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
    
    print(f"Compiled max_turns: {cfg.env.get('max_turns', 'NOT_SET')}")
    
    # Create environment manager
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(cfg.seed)
    
    env_ref = collector_env._env_ref
    print(f"Environment max_turns after config: {getattr(env_ref, 'max_turns', 'NOT_SET')}")
    
    # Test episode
    collector_env.reset({0: None})
    
    for step in range(8):  # More than max_turns
        legal_actions = getattr(env_ref, 'legal_actions', [])
        if not legal_actions:
            break
            
        action = legal_actions[0]
        print(f"Step {step + 1}: turn_count={getattr(env_ref.state, 'turn_count', 'N/A')}")
        
        timesteps = collector_env.step({0: action})
        timestep = timesteps[0]
        
        print(f"  Result: done={timestep.done}")
        
        if timestep.done:
            print(f"✅ Episode completed after {step + 1} steps!")
            collector_env.close()
            return True
    
    print("❌ Episode didn't complete even with max_turns=3")
    collector_env.close()
    return False

if __name__ == '__main__':
    success1 = test_environment_through_config()
    success2 = test_forced_short_episode()
    
    if success1 and success2:
        print("\n✅ Environment works correctly through config system")
    else:
        print("\n❌ Issue found with config system environment creation")
