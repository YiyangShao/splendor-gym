"""Training utilities for PPO Splendor training.

This module contains logging, plotting, checkpointing, and evaluation utilities
to keep the main training script clean and focused on core PPO logic.
"""

import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper, random_opponent
from splendor_gym.scripts.eval_suite import (
    eval_vs_opponent, 
    model_greedy_policy_from, 
    make_selfplay_env_with,
    greedy_opponent_v1,
    basic_priority_opponent,
)


@dataclass 
class TrainingHistory:
    """Container for training history data."""
    steps: List[int] = field(default_factory=list)
    wr_rand: List[float] = field(default_factory=list)
    wr_greedy1: List[float] = field(default_factory=list)
    wr_basic: List[float] = field(default_factory=list) 
    wr_self: List[float] = field(default_factory=list)
    turns_rand: List[float] = field(default_factory=list)
    turns_greedy1: List[float] = field(default_factory=list)
    turns_basic: List[float] = field(default_factory=list)
    turns_self: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    pol_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)


class TrainingLogger:
    """Handles all logging, plotting, and checkpointing for PPO training."""
    
    def __init__(self, log_dir: str, track: bool = False):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir) if track else None
        self.history = TrainingHistory()
        self.run_start_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def log_training_metrics(self, global_step: int, lr: float, policy_loss: float, 
                           value_loss: float, entropy: float, approx_kl: float):
        """Log training metrics to tensorboard."""
        if self.writer is None:
            return
            
        self.writer.add_scalar("charts/learning_rate", lr, global_step)
        self.writer.add_scalar("losses/policy_loss", policy_loss, global_step)
        self.writer.add_scalar("losses/value_loss", value_loss, global_step)
        self.writer.add_scalar("losses/entropy", entropy, global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        
    def log_evaluation_results(self, results: Dict[str, Dict], global_step: int):
        """Log evaluation results to tensorboard and update history."""
        if self.writer is None:
            return
            
        # Log all metrics
        for name, res in results.items():
            prefix = f"eval/win_rate_{name.replace('_', '')}"
            self.writer.add_scalar(prefix, res["win_rate"], global_step)
            self.writer.add_scalar(f"{prefix}_ci95", res["win_rate_ci95"], global_step)
            self.writer.add_scalar(f"eval/avg_turns_{name.replace('_', '')}", 
                                 res["avg_turns"], global_step)
            
        # Special metrics
        if "random" in results:
            self.writer.add_scalar("eval/draw_rate_random", 
                                 results["random"]["draws"] / max(1, results["random"]["n"]), 
                                 global_step)
        if "greedy_v1" in results:
            self.writer.add_scalar("eval/avg_prestige", 
                                 results["greedy_v1"]["avg_prestige"], global_step)
    
    def update_history(self, global_step: int, results: Dict[str, Dict], 
                      lr: float, policy_loss: float, value_loss: float, entropy: float):
        """Update training history."""
        self.history.steps.append(global_step)
        self.history.wr_rand.append(results.get("random", {}).get("win_rate", 0))
        self.history.wr_greedy1.append(results.get("greedy_v1", {}).get("win_rate", 0))
        self.history.wr_basic.append(results.get("basic", {}).get("win_rate", 0))
        self.history.wr_self.append(results.get("self_play", {}).get("win_rate", 0))
        self.history.turns_rand.append(results.get("random", {}).get("avg_turns", 0))
        self.history.turns_greedy1.append(results.get("greedy_v1", {}).get("avg_turns", 0))
        self.history.turns_basic.append(results.get("basic", {}).get("avg_turns", 0))
        self.history.turns_self.append(results.get("self_play", {}).get("avg_turns", 0))
        self.history.lr.append(lr)
        self.history.pol_loss.append(policy_loss)
        self.history.val_loss.append(value_loss)
        self.history.entropy.append(entropy)
    
    def create_summary_plot(self, global_step: int) -> bool:
        """Create and save summary training plot."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            
            # Win rates subplot
            ax = axes[0,0]
            ax.plot(self.history.steps, self.history.wr_rand, label="random")
            ax.plot(self.history.steps, self.history.wr_greedy1, label="greedy_v1")
            ax.plot(self.history.steps, self.history.wr_basic, label="basic_priority")
            ax.plot(self.history.steps, self.history.wr_self, label="self_play")
            ax.set_ylim(0, 1.0)
            ax.set_title("Win Rates")
            ax.set_xlabel("steps")
            ax.set_ylabel("win rate")
            ax.legend()
            
            # Average turns subplot  
            ax = axes[0,1]
            ax.plot(self.history.steps, self.history.turns_rand, label="random")
            ax.plot(self.history.steps, self.history.turns_greedy1, label="greedy_v1") 
            ax.plot(self.history.steps, self.history.turns_basic, label="basic_priority")
            ax.plot(self.history.steps, self.history.turns_self, label="self_play")
            ax.set_title("Avg Turns")
            ax.set_xlabel("steps")
            ax.set_ylabel("turns")
            ax.legend()
            
            # Add turn efficiency analysis
            if len(self.history.turns_rand) > 1:
                recent_turns = np.mean(self.history.turns_rand[-5:]) if len(self.history.turns_rand) >= 5 else self.history.turns_rand[-1]
                ax.axhline(y=recent_turns, color='red', linestyle='--', alpha=0.5, label=f'Recent avg: {recent_turns:.1f}')
            
            # Losses subplot
            ax = axes[1,0]
            x_loss = list(range(len(self.history.pol_loss)))
            ax.plot(x_loss, self.history.pol_loss, label="policy")
            ax.plot(x_loss, self.history.val_loss, label="value")
            ax.plot(x_loss, self.history.entropy, label="entropy")
            ax.set_title("Losses / Entropy")
            ax.set_xlabel("updates")
            ax.legend()
            
            # Learning rate subplot
            ax = axes[1,1]
            x_lr = list(range(len(self.history.lr)))
            ax.plot(x_lr, self.history.lr, label="lr")
            ax.set_title("Learning Rate")
            ax.set_xlabel("updates")
            
            # Overall formatting
            fig.suptitle(f"Summary @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save plot
            if self.writer is not None:
                self.writer.add_figure("eval/summary", fig, global_step)
            
            out_path_ts = os.path.join(self.log_dir, f"summary_{self.run_start_ts}.png")
            out_path_latest = os.path.join(self.log_dir, "summary.png")
            fig.savefig(out_path_ts)
            fig.savefig(out_path_latest)
            plt.close(fig)
            return True
            
        except Exception as e:
            print(f"[warn] plotting failed: {e}")
            return False


class CheckpointManager:
    """Manages model checkpointing and loading."""
    
    def __init__(self, log_dir: str, run_start_ts: str):
        self.log_dir = log_dir
        self.run_start_ts = run_start_ts
        
    def save_checkpoint(self, model: nn.Module, suffix: str = ""):
        """Save model checkpoint with timestamped and latest versions."""
        latest_path = os.path.join(self.log_dir, f"ppo_splendor_latest{suffix}.pt")
        ts_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ts_dir, exist_ok=True)
        ts_path = os.path.join(ts_dir, f"ppo_splendor_{self.run_start_ts}{suffix}.pt")
        
        torch.save(model.state_dict(), latest_path)
        torch.save(model.state_dict(), ts_path)
        return latest_path, ts_path


def make_env(seed: int, opponent_policy=None, opponent_supplier=None, 
             random_starts: bool = False):
    """Unified environment creation function."""
    # Default to random opponent if nothing specified
    if opponent_policy is None and opponent_supplier is None:
        opponent_policy = random_opponent
        
    def thunk():
        env = SplendorEnv(num_players=2)
        env = SelfPlayWrapper(
            env, 
            opponent_policy=opponent_policy or random_opponent,
            opponent_supplier=opponent_supplier,
            random_starts=random_starts
        )
        env.reset(seed=seed)
        return env
    return thunk


def run_evaluation_suite(agent: nn.Module, device: torch.device, rng: np.random.RandomState,
                        n_games: int, update_seed: int = 0) -> Dict[str, Dict]:
    """Run evaluation against all opponent types."""
    policy_eval = model_greedy_policy_from(agent, device=device)
    
    results = {}
    opponents = [
        ("random", random_opponent),
        ("greedy_v1", greedy_opponent_v1), 
        ("basic", basic_priority_opponent)
    ]
    
    for i, (name, opponent) in enumerate(opponents):
        env_fn = lambda: make_selfplay_env_with(opponent, int(rng.randint(1e9)))()
        results[name] = eval_vs_opponent(env_fn, policy_eval, n_games=n_games, 
                                       seed=update_seed + i)
    
    # Self-play evaluation
    opp_self = model_greedy_policy_from(agent, device=device)
    env_fn = lambda: make_selfplay_env_with(opp_self, int(rng.randint(1e9)))()
    results["self_play"] = eval_vs_opponent(env_fn, policy_eval, n_games=n_games,
                                          seed=update_seed + 3)
    
    return results


def frozen_policy_from(state_dict: dict, actor_critic_class, obs_dim: int, act_dim: int, device: torch.device):
    """Create a frozen policy from a state dict."""
    frozen = actor_critic_class(obs_dim, act_dim).to(device)
    frozen.load_state_dict(state_dict)
    frozen.eval()
    
    @torch.no_grad()
    def _policy(obs, info):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
        logits = frozen.actor(obs_t)
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
        return int(torch.argmax(logits, dim=-1).item())
    return _policy


def linear_lr_schedule(initial_lr: float, progress: float) -> float:
    """Linear learning rate schedule."""
    return initial_lr * progress
