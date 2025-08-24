import argparse
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper, random_opponent
from splendor_gym.engine.encode import OBSERVATION_DIM, TOTAL_ACTIONS
from splendor_gym.scripts.eval_suite import (
	make_selfplay_env,
	model_greedy_policy_from,
	eval_vs_opponent,
	greedy_opponent_v1,
)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
	# mask shape [B, A]; ensure at least one legal per row
	illegal = (mask < 0.5)
	masked_logits = logits.clone()
	# rows with at least one legal: apply -inf to illegal
	rows_any_legal = (~illegal).any(dim=1)
	if rows_any_legal.any():
		masked_logits[rows_any_legal] = masked_logits[rows_any_legal].masked_fill(
			illegal[rows_any_legal], float("-inf")
		)
	# rows with no legal actions: leave logits unmasked to avoid NaNs; env will handle no-legal at step
	return Categorical(logits=masked_logits)


class ActorCritic(nn.Module):
	def __init__(self, obs_dim: int, act_dim: int):
		super().__init__()
		self.critic = nn.Sequential(
			nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1)
		)
		self.actor = nn.Sequential(
			nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, act_dim)
		)

	def get_value(self, x: torch.Tensor) -> torch.Tensor:
		return self.critic(x)

	def get_action_and_value(self, x: torch.Tensor, mask: torch.Tensor, action: torch.Tensor | None = None):
		logits = self.actor(x)
		probs = masked_categorical(logits, mask)
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action), probs.entropy().mean(), self.critic(x)


def make_env(seed: int):
	def thunk():
		env = SplendorEnv(num_players=2)
		env = SelfPlayWrapper(env, opponent_policy=random_opponent)
		env.reset(seed=seed)
		return env
	return thunk


def linear_lr_schedule(initial_lr: float, progress: float) -> float:
	return initial_lr * progress


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--total-timesteps", type=int, default=1_000_000)
	parser.add_argument("--num-envs", type=int, default=16)
	parser.add_argument("--num-steps", type=int, default=128)
	parser.add_argument("--gamma", type=float, default=0.999)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--lr", type=float, default=2.5e-4)
	parser.add_argument("--ent-coef", type=float, default=0.03)
	parser.add_argument("--vf-coef", type=float, default=0.5)
	parser.add_argument("--clip-coef", type=float, default=0.2)
	parser.add_argument("--update-epochs", type=int, default=4)
	parser.add_argument("--minibatch-size", type=int, default=256)
	parser.add_argument("--save-path", type=str, default="runs/ppo_splendor.pt")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--track", action="store_true", help="log to tensorboard")
	parser.add_argument("--log-dir", type=str, default="runs/ppo_splendor")
	parser.add_argument("--eval-every-updates", type=int, default=10)
	parser.add_argument("--eval-games", type=int, default=400)
	parser.add_argument("--lr-anneal", action="store_true")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Seeding
	rng = np.random.RandomState(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	num_envs = args.num_envs
	num_steps = args.num_steps
	envs = gym.vector.SyncVectorEnv([make_env(int(rng.randint(1e9))) for _ in range(num_envs)])

	obs = np.zeros((num_envs, OBSERVATION_DIM), dtype=np.int32)
	masks = np.zeros((num_envs, TOTAL_ACTIONS), dtype=np.int8)
	obs_t, info = envs.reset()
	obs[:] = obs_t
	if isinstance(info, dict) and "action_mask" in info:
		am = info["action_mask"]
		masks[:] = am if isinstance(am, np.ndarray) and am.shape == (num_envs, TOTAL_ACTIONS) else np.stack([am[i] for i in range(num_envs)], axis=0)
	else:
		for i in range(num_envs):
			masks[i] = info[i]["action_mask"]

	agent = ActorCritic(OBSERVATION_DIM, TOTAL_ACTIONS).to(device)
	optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

	writer = SummaryWriter(args.log_dir) if args.track else None

	num_updates = args.total_timesteps // (num_envs * num_steps)
	global_step = 0
	best_wr_greedy = -1.0

	for update in range(num_updates):
		# LR annealing
		if args.lr_anneal:
			progress = 1.0 - (update / max(1, num_updates - 1))
			lr = linear_lr_schedule(args.lr, progress)
			for pg in optimizer.param_groups:
				pg["lr"] = lr

		# Rollout storage
		obs_buf = []
		masks_buf = []
		actions_buf = []
		logprobs_buf = []
		rewards_buf = []
		values_buf = []
		terminals_buf = []

		for step in range(num_steps):
			obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
			mask_tensor = torch.tensor(masks, dtype=torch.float32, device=device)
			with torch.no_grad():
				action, logprob, entropy, value = agent.get_action_and_value(obs_tensor, mask_tensor)
			actions = action.cpu().numpy()
			next_obs, rewards, terms, truncs, infos = envs.step(actions)
			if isinstance(infos, dict) and "action_mask" in infos:
				am = infos["action_mask"]
				next_masks = am if isinstance(am, np.ndarray) and am.shape == (num_envs, TOTAL_ACTIONS) else np.stack([am[i] for i in range(num_envs)], axis=0)
			else:
				next_masks = np.zeros_like(masks)
				for i in range(num_envs):
					next_masks[i] = infos[i]["action_mask"]

			obs_buf.append(obs.copy())
			masks_buf.append(masks.copy())
			actions_buf.append(actions.copy())
			logprobs_buf.append(logprob.detach().cpu().numpy())
			values_buf.append(value.detach().cpu().numpy())
			rewards_buf.append(rewards.copy())
			terminals_buf.append(terms.copy())

			obs = next_obs
			masks = next_masks
			global_step += num_envs

		# Compute returns/advantages (GAE)
		with torch.no_grad():
			obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
			last_values = agent.get_value(obs_tensor).detach().cpu().numpy().squeeze(-1)

		rewards_arr = np.array(rewards_buf)  # [T, N]
		values_arr = np.array(values_buf).squeeze(-1)  # [T, N]
		term_arr = np.array(terminals_buf)  # [T, N]
		advantages = np.zeros_like(rewards_arr)
		lastgaelam = np.zeros(num_envs)
		for t in reversed(range(num_steps)):
			nextnonterminal = 1.0 - term_arr[t]
			nextvalues = last_values if t == num_steps - 1 else values_arr[t + 1]
			delta = rewards_arr[t] + args.gamma * nextvalues * nextnonterminal - values_arr[t]
			advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
		returns = advantages + values_arr

		# Flatten
		b_obs = torch.tensor(np.concatenate(obs_buf, axis=0), dtype=torch.float32, device=device)
		b_masks = torch.tensor(np.concatenate(masks_buf, axis=0), dtype=torch.float32, device=device)
		b_actions = torch.tensor(np.concatenate(actions_buf, axis=0), dtype=torch.int64, device=device)
		b_logprobs = torch.tensor(np.concatenate(logprobs_buf, axis=0), dtype=torch.float32, device=device)
		b_returns = torch.tensor(np.concatenate(returns, axis=0), dtype=torch.float32, device=device)
		b_values = torch.tensor(np.concatenate(values_arr, axis=0), dtype=torch.float32, device=device)
		b_advantages = torch.tensor(np.concatenate(advantages, axis=0), dtype=torch.float32, device=device)
		b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

		# PPO update
		batch_size = b_obs.shape[0]
		minibatch_size = min(args.minibatch_size, batch_size)
		for epoch in range(args.update_epochs):
			idxs = torch.randperm(batch_size, device=device)
			for start in range(0, batch_size, minibatch_size):
				mb_idx = idxs[start:start + minibatch_size]
				new_action, new_logprob, entropy, new_value = agent.get_action_and_value(
					b_obs[mb_idx], b_masks[mb_idx], b_actions[mb_idx]
				)
				ratio = (new_logprob - b_logprobs[mb_idx]).exp()
				mb_adv = b_advantages[mb_idx]
				clip_adv = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_adv
				policy_loss = -torch.min(ratio * mb_adv, clip_adv).mean()
				value_loss = 0.5 * (new_value.squeeze(-1) - b_returns[mb_idx]).pow(2).mean()
				entropy_loss = -entropy.mean()
				loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * (-entropy_loss)

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
				optimizer.step()

		# Logging
		if writer is not None and (update + 1) % 1 == 0:
			writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
			writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
			writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
			writer.add_scalar("losses/entropy", (-entropy_loss).item(), global_step)

		# Periodic evaluation
		if (update + 1) % args.eval_every_updates == 0:
			policy_eval = model_greedy_policy_from(agent, device=device)
			res_rand = eval_vs_opponent(lambda: make_selfplay_env(int(rng.randint(1e9)))(), policy_eval, n_games=args.eval_games, seed=update)
			res_greedy = eval_vs_opponent(lambda: make_selfplay_env(int(rng.randint(1e9)))(), policy_eval, n_games=args.eval_games, seed=update+1)
			if writer is not None:
				writer.add_scalar("eval/win_rate_random", res_rand["win_rate"], global_step)
				writer.add_scalar("eval/win_rate_greedy_v1", res_greedy["win_rate"], global_step)
				writer.add_scalar("eval/draw_rate_random", res_rand["draws"] / res_rand["n"], global_step)
				writer.add_scalar("eval/avg_turns", res_greedy["avg_turns"], global_step)
				writer.add_scalar("eval/avg_prestige", res_greedy["avg_prestige"], global_step)
			# Best checkpoint gating
			if res_greedy["win_rate"] > best_wr_greedy:
				best_wr_greedy = res_greedy["win_rate"]
				best_path = os.path.join(os.path.dirname(args.save_path), "ppo_splendor_best.pt")
				os.makedirs(os.path.dirname(best_path), exist_ok=True)
				torch.save(agent.state_dict(), best_path)
				print(f"[eval] new best vs greedy_v1: {best_wr_greedy:.3f} saved {best_path}")

		if (update + 1) % 10 == 0:
			print(f"update={update+1}/{num_updates}")

	# Save final
	os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
	torch.save(agent.state_dict(), args.save_path)
	print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
	main() 