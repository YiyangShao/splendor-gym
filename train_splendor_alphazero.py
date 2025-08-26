#!/usr/bin/env python3
"""
Splendor AlphaZero Training Script

This script provides a convenient way to train AlphaZero on Splendor using LightZero.

Usage:
    python train_splendor_alphazero.py
    
    # Or with custom parameters:
    python train_splendor_alphazero.py --max_env_step 500000 --exp_name my_splendor_run
"""

import os
import sys
import argparse
from pathlib import Path

# Add LightZero to path
LIGHTZERO_PATH = Path(__file__).parent / 'LightZero'
sys.path.insert(0, str(LIGHTZERO_PATH))

def main():
    parser = argparse.ArgumentParser(description='Train Splendor AlphaZero')
    parser.add_argument('--max_env_step', type=int, default=1000000,
                        help='Maximum environment steps (default: 1M)')
    parser.add_argument('--exp_name', type=str, default='splendor_alphazero_experiment',
                        help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_simulations', type=int, default=200,
                        help='Number of MCTS simulations per action')
    parser.add_argument('--collector_env_num', type=int, default=8,
                        help='Number of collector environments')
    parser.add_argument('--evaluator_env_num', type=int, default=5,
                        help='Number of evaluator environments')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='Evaluation frequency (in steps)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SPLENDOR ALPHAZERO TRAINING")
    print("="*60)
    print(f"Experiment name: {args.exp_name}")
    print(f"Max environment steps: {args.max_env_step:,}")
    print(f"MCTS simulations: {args.num_simulations}")
    print(f"Collector environments: {args.collector_env_num}")
    print(f"Evaluator environments: {args.evaluator_env_num}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Evaluation frequency: {args.eval_freq}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Import LightZero components
    from lzero.entry import train_alphazero
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    from easydict import EasyDict
    import copy
    
    # Update config with command line arguments
    config = EasyDict(copy.deepcopy(splendor_alphazero_config))
    config.exp_name = f'data_az_ctree/{args.exp_name}_seed{args.seed}'
    
    # Update environment config
    config.env.collector_env_num = args.collector_env_num
    config.env.evaluator_env_num = args.evaluator_env_num
    config.env.n_evaluator_episode = args.evaluator_env_num
    
    # Update policy config
    config.policy.collector_env_num = args.collector_env_num
    config.policy.evaluator_env_num = args.evaluator_env_num
    config.policy.batch_size = args.batch_size
    config.policy.learning_rate = args.learning_rate
    config.policy.eval_freq = args.eval_freq
    config.policy.mcts.num_simulations = args.num_simulations
    
    print("Starting training...")
    print("Note: Training logs and checkpoints will be saved in the 'data_az_ctree' directory")
    print("You can monitor progress using tensorboard:")
    print(f"  tensorboard --logdir data_az_ctree/{args.exp_name}_seed{args.seed}")
    print()
    
    try:
        # Start training
        train_alphazero([config, splendor_alphazero_create_config], 
                       seed=args.seed, 
                       max_env_step=args.max_env_step)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    else:
        print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
