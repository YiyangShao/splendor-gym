"""
Enhanced PPO training script for Splendor with improved architecture and hyperparameters.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from enhanced_splendor_env import EnhancedSplendorEnv, ENHANCED_OBS_DIM
from splendor_gym.engine.encode import TOTAL_ACTIONS
from splendor_gym.scripts.eval_suite import (
    model_greedy_policy_from,
    random_opponent,
    greedy_opponent_v1,
    basic_priority_opponent,
)
from training_utils import (
    TrainingLogger,
    CheckpointManager,
    run_evaluation_suite,
    frozen_policy_from,
    linear_lr_schedule,
)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Create categorical distribution with action masking."""
    illegal = (mask < 0.5)
    masked_logits = logits.clone()
    rows_any_legal = (~illegal).any(dim=1)
    if rows_any_legal.any():
        masked_logits[rows_any_legal] = masked_logits[rows_any_legal].masked_fill(
            illegal[rows_any_legal], float("-inf")
        )
    return Categorical(logits=masked_logits)


class EnhancedActorCritic(nn.Module):
    """Enhanced Actor-Critic with larger capacity and modern techniques."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 512):
        super().__init__()
        
        # Shared backbone with residual connections
        self.shared_backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), 
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        
        # Separate heads with residual connection
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, act_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01 if module == self.actor_head[-1] else 1.0)
            module.bias.data.fill_(0.0)
    
    def forward(self, x: torch.Tensor):
        features = self.shared_backbone(x)
        return self.actor_head(features), self.critic_head(features)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_backbone(x)
        return self.critic_head(features)
    
    def get_action_and_value(self, x: torch.Tensor, mask: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self(x)
        probs = masked_categorical(logits, mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


def make_enhanced_env(seed: int, opponent_supplier=None, opponent_policy=None):
    """Create enhanced Splendor environment.""" 
    def thunk():
        env = EnhancedSplendorEnv()
        if opponent_supplier is not None or opponent_policy is not None:
            from splendor_gym.wrappers.selfplay import SelfPlayWrapper
            env = SelfPlayWrapper(
                env,
                opponent_policy=opponent_policy or random_opponent,
                opponent_supplier=opponent_supplier,
                random_starts=True
            )
        env.reset(seed=seed)
        return env
    return thunk


def main():
    parser = argparse.ArgumentParser()
    # Enhanced hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)  # Increased
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=256)  # Increased for longer horizons
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)  # Slightly higher
    parser.add_argument("--ent-coef", type=float, default=0.1)  # Higher for exploration
    parser.add_argument("--ent-coef-final", type=float, default=0.01)  # Decay target
    parser.add_argument("--vf-coef", type=float, default=0.25)  # Lower to prevent value domination
    parser.add_argument("--clip-coef", type=float, default=0.3)  # Less conservative
    parser.add_argument("--vclip", type=float, default=0.3)  # Match clip-coef
    parser.add_argument("--update-epochs", type=int, default=6)  # More updates
    parser.add_argument("--minibatch-size", type=int, default=512)  # Larger minibatches
    parser.add_argument("--hidden-size", type=int, default=512)  # Larger network
    parser.add_argument("--target-kl", type=float, default=0.03)  # Less conservative
    
    # Self-play parameters
    parser.add_argument("--p-current", type=float, default=0.5)  # More self-play exposure
    parser.add_argument("--pool-size", type=int, default=16)  # Larger diversity
    parser.add_argument("--snapshot-every-updates", type=int, default=10)
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track", action="store_true", help="log to tensorboard")
    parser.add_argument("--log-dir", type=str, default="runs/ppo_enhanced")
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=400)
    parser.add_argument("--lr-anneal", action="store_true", default=True)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using: {device}")
    if device.type == "cuda":
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    
    # Seeding
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Self-play opponent pool
    pool: list[dict] = []
    
    def opponent_supplier():
        if len(pool) == 0 or np.random.rand() < args.p_current:
            return model_greedy_policy_from(agent, device=device)
        idx = int(np.random.randint(0, len(pool)))
        return frozen_policy_from(pool[idx], EnhancedActorCritic, ENHANCED_OBS_DIM, TOTAL_ACTIONS, device)
    
    # Create enhanced agent
    agent = EnhancedActorCritic(ENHANCED_OBS_DIM, TOTAL_ACTIONS, args.hidden_size).to(device)
    
    # Enhanced optimizer with weight decay
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-4)
    
    # Create environments with enhanced environment
    envs = gym.vector.SyncVectorEnv([
        make_enhanced_env(int(rng.randint(1e9)), opponent_supplier=opponent_supplier)
        for _ in range(args.num_envs)
    ])
    
    # Initialize observation buffers 
    obs = np.zeros((args.num_envs, ENHANCED_OBS_DIM), dtype=np.int32)
    masks = np.zeros((args.num_envs, TOTAL_ACTIONS), dtype=np.int8)
    
    obs_t, info = envs.reset()
    obs[:] = obs_t
    for i in range(args.num_envs):
        masks[i] = info[i]["action_mask"]
    
    # Training setup
    logger = TrainingLogger(args.log_dir, track=args.track)
    checkpoint_manager = CheckpointManager(args.log_dir, logger.run_start_ts)
    
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    global_step = 0
    
    # Enhanced diagnostic tracking
    value_losses = []
    policy_losses = []
    entropies = []
    explained_variances = []
    
    print(f"Starting enhanced training: {num_updates} updates, {args.total_timesteps} total timesteps")
    print(f"Enhanced observation dim: {ENHANCED_OBS_DIM}, Hidden size: {args.hidden_size}")
    
    for update in range(num_updates):
        # Dynamic learning rate annealing
        if args.lr_anneal:
            progress = 1.0 - (update / max(1, num_updates - 1))
            lr = linear_lr_schedule(args.lr, progress) 
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        
        # Dynamic entropy coefficient annealing
        progress_ent = update / max(1, num_updates - 1)
        ent_coef_now = args.ent_coef + (args.ent_coef_final - args.ent_coef) * progress_ent
        
        # Rollout collection (same as before but with enhanced observations)
        obs_buf, masks_buf, actions_buf, logprobs_buf = [], [], [], []
        rewards_buf, values_buf, terminals_buf = [], [], []
        
        for step in range(args.num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            mask_tensor = torch.tensor(masks, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(obs_tensor, mask_tensor)
            
            actions = action.cpu().numpy()
            next_obs, rewards, terms, truncs, infos = envs.step(actions)
            
            # Update masks from info
            for i in range(args.num_envs):
                masks[i] = infos[i]["action_mask"]
            
            # Store rollout data
            obs_buf.append(obs.copy())
            masks_buf.append(masks.copy())
            actions_buf.append(actions.copy())
            logprobs_buf.append(logprob.detach().cpu().numpy())
            values_buf.append(value.detach().cpu().numpy())
            rewards_buf.append(rewards.copy())
            terminals_buf.append(terms.copy())
            
            obs = next_obs
            global_step += args.num_envs
        
        # GAE calculation (same as before)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            last_values = agent.get_value(obs_tensor).detach().cpu().numpy().squeeze(-1)
        
        # Calculate advantages and returns
        rewards_arr = np.array(rewards_buf)
        values_arr = np.array(values_buf).squeeze(-1)
        term_arr = np.array(terminals_buf)
        advantages = np.zeros_like(rewards_arr)
        lastgaelam = np.zeros(args.num_envs)
        
        for t in reversed(range(args.num_steps)):
            nextnonterminal = 1.0 - term_arr[t]
            nextvalues = last_values if t == args.num_steps - 1 else values_arr[t + 1]
            delta = rewards_arr[t] + args.gamma * nextvalues * nextnonterminal - values_arr[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values_arr
        
        # Flatten batch data
        b_obs = torch.tensor(np.concatenate(obs_buf, axis=0), dtype=torch.float32, device=device)
        b_masks = torch.tensor(np.concatenate(masks_buf, axis=0), dtype=torch.float32, device=device) 
        b_actions = torch.tensor(np.concatenate(actions_buf, axis=0), dtype=torch.int64, device=device)
        b_logprobs = torch.tensor(np.concatenate(logprobs_buf, axis=0), dtype=torch.float32, device=device)
        b_returns = torch.tensor(np.concatenate(returns, axis=0), dtype=torch.float32, device=device)
        b_values = torch.tensor(np.concatenate(values_arr, axis=0), dtype=torch.float32, device=device)
        b_advantages = torch.tensor(np.concatenate(advantages, axis=0), dtype=torch.float32, device=device)
        
        # Advantage normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Enhanced PPO updates with better diagnostics
        batch_size = b_obs.shape[0]
        minibatch_size = min(args.minibatch_size, batch_size)
        
        for epoch in range(args.update_epochs):
            idxs = torch.randperm(batch_size, device=device)
            
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]
                
                new_action, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_idx], b_masks[mb_idx], b_actions[mb_idx]
                )
                
                # Policy loss with enhanced clipping
                ratio = (new_logprob - b_logprobs[mb_idx]).exp()
                mb_adv = b_advantages[mb_idx]
                clip_adv = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_adv
                policy_loss = -torch.min(ratio * mb_adv, clip_adv).mean()
                
                # Value loss with clipping
                v_pred = new_value.squeeze(-1)
                v_pred_clipped = b_values[mb_idx] + torch.clamp(
                    v_pred - b_values[mb_idx], -args.vclip, args.vclip
                )
                v_loss_unclipped = (v_pred - b_returns[mb_idx]).pow(2)
                v_loss_clipped = (v_pred_clipped - b_returns[mb_idx]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Combined loss
                loss = policy_loss + args.vf_coef * value_loss + ent_coef_now * entropy_loss
                
                # Gradient update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                # Early stopping on KL divergence
                approx_kl = (b_logprobs[mb_idx] - new_logprob).mean().detach().cpu().item()
                if args.target_kl > 0 and approx_kl > args.target_kl:
                    break
        
        # Enhanced diagnostics
        with torch.no_grad():
            # Calculate explained variance
            y_true = b_returns.cpu().numpy()
            y_pred = agent.get_value(b_obs).squeeze(-1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            explained_variances.append(explained_var)
        
        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())
        entropies.append(entropy.mean().item())
        
        # Pool management
        if (update + 1) % args.snapshot_every_updates == 0:
            pool.append(agent.state_dict())
            if len(pool) > args.pool_size:
                pool.pop(0)
        
        # Checkpoint saving
        checkpoint_manager.save_checkpoint(agent)
        
        # Enhanced logging every update
        if logger.writer is not None:
            logger.writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
            logger.writer.add_scalar("train/value_loss", value_loss.item(), global_step)
            logger.writer.add_scalar("train/entropy", entropy.mean().item(), global_step)
            logger.writer.add_scalar("train/explained_variance", explained_var, global_step)
            logger.writer.add_scalar("train/entropy_coef", ent_coef_now, global_step)
            logger.writer.add_scalar("train/approx_kl", approx_kl, global_step)
            logger.writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            
            # Reward statistics
            avg_reward = np.mean(rewards_arr)
            logger.writer.add_scalar("train/avg_reward", avg_reward, global_step)
        
        # Periodic evaluation with enhanced metrics
        if (update + 1) % args.eval_every_updates == 0:
            print(f"\n[Eval] Update {update + 1}/{num_updates}")
            print(f"  Policy Loss: {policy_loss.item():.4f}")
            print(f"  Value Loss: {value_loss.item():.4f}") 
            print(f"  Entropy: {entropy.mean().item():.4f}")
            print(f"  Explained Var: {explained_var:.4f}")
            print(f"  Avg Reward: {np.mean(rewards_arr):.4f}")
            
            # Run evaluation (reuse from training_utils)
            try:
                results = run_evaluation_suite(agent, device, rng, args.eval_games, update)
                logger.log_evaluation_results(results, global_step)
                
                print(f"  Win vs Random: {results['random']['win_rate']:.3f}")
                print(f"  Win vs Basic: {results['basic']['win_rate']:.3f}")
                print(f"  Pool size: {len(pool)}")
            except Exception as e:
                print(f"  Evaluation failed: {e}")
        
        if (update + 1) % 25 == 0:
            print(f"Update {update + 1}/{num_updates}, Step {global_step}, Entropy: {entropy.mean().item():.3f}")
    
    # Final save
    final_path = os.path.join(args.log_dir, f"ppo_enhanced_final_{logger.run_start_ts}.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
