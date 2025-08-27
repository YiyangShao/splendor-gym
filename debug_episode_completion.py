#!/usr/bin/env python3
"""
Debug why episodes never complete
"""

import sys
sys.path.insert(0, 'LightZero')

def test_episode_completion():
    """Test if episodes complete properly"""
    
    print("=== DEBUG: Episode Completion ===")
    
    from LightZero.zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
    
    # Create environment directly
    env = SplendorLightZeroEnv()
    print(f"Environment max_turns: {env.max_turns}")
    
    # Reset environment
    obs = env.reset()
    print(f"Initial state: to_play={obs['to_play']}, turn_count={env.state.turn_count}")
    
    step_count = 0
    max_steps = 100  # Safety limit
    
    print("\nRunning episode step by step...")
    
    while step_count < max_steps:
        # Check current game state
        legal_actions = env.legal_actions
        if not legal_actions:
            print(f"❌ No legal actions at step {step_count}")
            break
            
        # Take first legal action
        action = legal_actions[0]
        
        print(f"Step {step_count + 1}:")
        print(f"  - Turn count: {env.state.turn_count}")
        print(f"  - Current player: {env.state.to_play}")
        print(f"  - Game over: {env.state.game_over}")
        print(f"  - Turn limit reached: {env.state.turn_limit_reached}")
        print(f"  - Action: {action}")
        
        # Take step
        timestep = env.step(action)
        step_count += 1
        
        print(f"  - After step: done={timestep.done}, reward={timestep.reward}")
        print(f"  - New turn count: {env.state.turn_count}")
        print(f"  - New game over: {env.state.game_over}")
        
        if timestep.done:
            print(f"✅ Episode completed after {step_count} steps!")
            print(f"   - Final scores: {[p.prestige for p in env.state.players]}")
            print(f"   - Turn limit reached: {env.state.turn_limit_reached}")
            print(f"   - eval_episode_return: {timestep.info.get('eval_episode_return', 'NOT_SET')}")
            return True
            
        # Check if we're hitting turn limit
        if env.state.turn_count >= env.max_turns:
            print(f"⚠️  Turn limit reached but episode not done!")
            print(f"   - Turn count: {env.state.turn_count}")
            print(f"   - Max turns: {env.max_turns}")
            print(f"   - Game over: {env.state.game_over}")
            print(f"   - Turn limit reached: {env.state.turn_limit_reached}")
            break
    
    print(f"❌ Episode didn't complete after {step_count} steps")
    print(f"   - Final turn count: {env.state.turn_count}")
    print(f"   - Max turns: {env.max_turns}")
    print(f"   - Game over: {env.state.game_over}")
    return False

def test_with_different_max_turns():
    """Test with very low max_turns to force completion"""
    
    print("\n=== Testing with max_turns=5 ===")
    
    from LightZero.zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
    
    env = SplendorLightZeroEnv()
    env.max_turns = 5  # Very short
    
    obs = env.reset()
    print(f"Environment max_turns set to: {env.max_turns}")
    
    for step in range(10):  # More than max_turns
        legal_actions = env.legal_actions
        if not legal_actions:
            break
            
        action = legal_actions[0]
        timestep = env.step(action)
        
        print(f"Step {step + 1}: turn_count={env.state.turn_count}, done={timestep.done}")
        
        if timestep.done:
            print(f"✅ Episode completed at step {step + 1}!")
            return True
    
    print("❌ Episode still didn't complete with max_turns=5")
    return False

if __name__ == '__main__':
    success1 = test_episode_completion()
    success2 = test_with_different_max_turns()
    
    if success1 or success2:
        print("\n✅ Episode completion works!")
    else:
        print("\n❌ Episode completion is broken - this is why collection hangs!")
