#!/usr/bin/env python3
"""
Splendor AlphaZero Evaluation Script

This script provides a convenient way to evaluate trained AlphaZero models on Splendor.

Usage:
    python eval_splendor_alphazero.py --model_path path/to/model.pth.tar
"""

import os
import sys
import argparse
from pathlib import Path

# Add LightZero to path
LIGHTZERO_PATH = Path(__file__).parent / 'LightZero'
sys.path.insert(0, str(LIGHTZERO_PATH))

def main():
    parser = argparse.ArgumentParser(description='Evaluate Splendor AlphaZero')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth.tar)')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--num_simulations', type=int, default=400,
                        help='Number of MCTS simulations for evaluation (default: 400)')
    parser.add_argument('--exp_name', type=str, default='splendor_eval',
                        help='Experiment name for evaluation results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for evaluation')
    parser.add_argument('--vs_bot', action='store_true',
                        help='Evaluate against noble strategy bot')
    parser.add_argument('--vs_random', action='store_true',
                        help='Evaluate against random player')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        return 1
    
    print("="*60)
    print("SPLENDOR ALPHAZERO EVALUATION")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"MCTS simulations: {args.num_simulations}")
    print(f"Experiment name: {args.exp_name}")
    print(f"Random seed: {args.seed}")
    
    if args.vs_bot:
        print("Evaluation mode: vs Noble Strategy Bot")
    elif args.vs_random:
        print("Evaluation mode: vs Random Player")
    else:
        print("Evaluation mode: Self-play")
    print("="*60)
    
    # Import LightZero components
    from lzero.entry import eval_alphazero
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import (
        splendor_alphazero_config, splendor_alphazero_create_config
    )
    
    # Update config for evaluation
    config = splendor_alphazero_config.copy()
    config.exp_name = f'eval_{args.exp_name}_seed{args.seed}'
    
    # Update evaluation settings
    config.env.evaluator_env_num = min(args.num_episodes, 10)  # Don't create too many envs
    config.env.n_evaluator_episode = args.num_episodes
    config.policy.evaluator_env_num = config.env.evaluator_env_num
    config.policy.mcts.num_simulations = args.num_simulations
    
    # Set evaluation mode
    if args.vs_bot:
        config.env.battle_mode = 'play_with_bot_mode'
        config.env.bot_action_type = 'noble_strategy'
    elif args.vs_random:
        config.env.prob_random_agent = 1.0
    
    print("Starting evaluation...")
    print("Results will be logged in the evaluation output directory")
    print()
    
    try:
        # Start evaluation
        eval_alphazero([config, splendor_alphazero_create_config], 
                      seed=args.seed,
                      load_path=args.model_path)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        raise
    else:
        print("\nEvaluation completed successfully!")

if __name__ == '__main__':
    main()
