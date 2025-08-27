#!/usr/bin/env python3
"""
Working Splendor AlphaZero Training Script

This version includes all the fixes we've discovered and optimized settings
for reasonable training times.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add LightZero to path
LIGHTZERO_PATH = Path(__file__).parent / 'LightZero'
sys.path.insert(0, str(LIGHTZERO_PATH))

def create_optimized_config(args):
    """Create config with all our fixes and optimizations"""
    
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    from easydict import EasyDict
    import copy
    
    # Start with base config
    config = EasyDict(copy.deepcopy(splendor_alphazero_config))
    create_config = EasyDict(copy.deepcopy(splendor_alphazero_create_config))
    
    # CRITICAL FIXES:
    # 1. Use 'base' environment manager (not 'subprocess') to avoid caching issues
    config.env.manager.type = 'base'
    
    # 2. Set reasonable max_turns to ensure episode completion
    config.env.max_turns = args.max_turns
    
    # 3. Optimize MCTS simulations for reasonable speed
    config.policy.mcts.num_simulations = args.num_simulations
    
    # 4. Environment settings
    config.env.collector_env_num = args.collector_env_num
    config.env.evaluator_env_num = args.evaluator_env_num
    config.env.n_evaluator_episode = args.evaluator_env_num
    
    # 5. Policy settings
    config.policy.collector_env_num = args.collector_env_num
    config.policy.evaluator_env_num = args.evaluator_env_num
    config.policy.batch_size = args.batch_size
    config.policy.learning_rate = args.learning_rate
    config.policy.eval_freq = args.eval_freq
    
    # 6. Training settings
    config.exp_name = f'data_az_ctree/{args.exp_name}_seed{args.seed}'
    
    # 7. Use CPU for faster startup (can be changed to GPU)
    config.policy.device = args.device
    config.policy.cuda = (args.device != 'cpu')
    
    return config, create_config

def estimate_training_times(config):
    """Provide realistic time estimates based on config"""
    
    # Based on our testing:
    # - Environment episode: ~3-20s (depends on max_turns and strategy)
    # - Policy forward: ~0.05-0.2s per step (depends on MCTS sims)
    # - MCTS scaling: roughly linear with num_simulations
    
    max_turns = config.env.max_turns
    num_simulations = config.policy.mcts.num_simulations
    collector_envs = config.env.collector_env_num
    
    # Estimate episode length (turns = actions from both players)
    estimated_episode_turns = min(max_turns, 60)  # Typical game length
    
    # Time per MCTS forward pass
    mcts_time_per_step = 0.01 + (num_simulations * 0.003)  # Base + scaling
    
    # Time per episode
    episode_time = estimated_episode_turns * mcts_time_per_step
    
    # Collection time per episode
    collection_time_per_episode = episode_time + 2  # +overhead
    
    # Training iteration time
    episodes_per_iteration = config.policy.get('collect_n_episode', 8)
    iteration_time = (episodes_per_iteration / collector_envs) * collection_time_per_episode
    
    # Learning step time (roughly)
    learning_time_per_iteration = 0.5  # seconds
    
    total_iteration_time = iteration_time + learning_time_per_iteration
    
    print(f"\n📊 TRAINING TIME ESTIMATES:")
    print(f"=" * 50)
    print(f"Episode Settings:")
    print(f"  - Max turns: {max_turns}")
    print(f"  - Estimated episode turns: {estimated_episode_turns}")
    print(f"  - MCTS simulations: {num_simulations}")
    print(f"  - Collector environments: {collector_envs}")
    print(f"")
    print(f"Time Estimates:")
    print(f"  - MCTS forward pass: ~{mcts_time_per_step:.3f}s")
    print(f"  - Single episode: ~{collection_time_per_episode:.1f}s")
    print(f"  - Training iteration: ~{total_iteration_time:.1f}s")
    print(f"")
    print(f"Full Training Estimates:")
    print(f"  - 100 iterations: ~{(total_iteration_time * 100 / 60):.1f} minutes")
    print(f"  - 1,000 iterations: ~{(total_iteration_time * 1000 / 3600):.1f} hours")
    print(f"  - 10,000 iterations: ~{(total_iteration_time * 10000 / 3600):.1f} hours")
    print(f"=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Working Splendor AlphaZero Training')
    
    # Key parameters for performance tuning
    parser.add_argument('--max_env_step', type=int, default=50000,
                        help='Maximum environment steps (default: 50K for quick testing)')
    parser.add_argument('--exp_name', type=str, default='working_splendor',
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Performance critical parameters
    parser.add_argument('--num_simulations', type=int, default=25,
                        help='MCTS simulations (25=fast, 50=medium, 200=slow)')
    parser.add_argument('--max_turns', type=int, default=50,
                        help='Max turns per game (50=medium, 100=long)')
    parser.add_argument('--collector_env_num', type=int, default=1,
                        help='Collector environments (start with 1)')
    parser.add_argument('--evaluator_env_num', type=int, default=1,
                        help='Evaluator environments (start with 1)')
    
    # Training parameters  
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--eval_freq', type=int, default=2000, help='Evaluation frequency')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WORKING SPLENDOR ALPHAZERO TRAINING")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Max env steps: {args.max_env_step:,}")
    print(f"MCTS simulations: {args.num_simulations}")
    print(f"Max turns per game: {args.max_turns}")
    print(f"Environments: {args.collector_env_num} collector, {args.evaluator_env_num} evaluator")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    # Create optimized config
    config, create_config = create_optimized_config(args)
    
    # Show time estimates
    estimate_training_times(config)
    
    print(f"\n🚀 Starting training...")
    print(f"Logs: data_az_ctree/{args.exp_name}_seed{args.seed}")
    print(f"Tensorboard: tensorboard --logdir data_az_ctree/{args.exp_name}_seed{args.seed}")
    print(f"")
    
    try:
        start_time = time.time()
        
        # Import and start training
        from lzero.entry import train_alphazero
        
        train_alphazero(
            [config, create_config], 
            seed=args.seed, 
            max_env_step=args.max_env_step
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed:.1f}s!")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n⏹️  Training interrupted after {elapsed:.1f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Training failed after {elapsed:.1f}s: {e}")
        raise

if __name__ == '__main__':
    main()
