import numpy as np
import torch
from typing import Callable, Dict, Any

from splendor_gym.envs import SplendorEnv
from splendor_gym.wrappers.selfplay import SelfPlayWrapper, random_opponent
from splendor_gym.engine.encode import OBSERVATION_DIM, TOTAL_ACTIONS


def greedy_opponent_v1(obs, info):
	legal = np.flatnonzero(info["action_mask"])
	if len(legal) == 0:
		return 0
	# Buy visible/reserved preferred
	buys = [a for a in legal if (15 <= a <= 26) or (42 <= a <= 44)]
	if buys:
		return int(buys[0])
	# Take-2
	take2 = [a for a in legal if 10 <= a <= 14]
	if take2:
		return int(take2[0])
	# Take-3 (including reduced-color)
	take3 = [a for a in legal if 0 <= a <= 9]
	if take3:
		return int(take3[0])
	# Reserve
	res = [a for a in legal if 27 <= a <= 41]
	if res:
		return int(res[0])
	return int(legal[0])


@torch.no_grad()
def model_greedy_policy_from(model: torch.nn.Module, device: str = "cpu") -> Callable[[np.ndarray, Dict[str, Any]], int]:
	model.eval()
	def _policy(obs, info):
		obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
		mask = torch.tensor(info["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
		logits = model.actor(obs_t)
		logits = logits.masked_fill(mask < 0.5, float("-inf"))
		action = torch.argmax(logits, dim=-1).item()
		return int(action)
	return _policy


def make_selfplay_env(seed: int):
	def thunk():
		env = SplendorEnv(num_players=2)
		env = SelfPlayWrapper(env, opponent_policy=random_opponent)
		env.reset(seed=seed)
		return env
	return thunk


def eval_vs_opponent(make_env: Callable[[], Any], model_policy: Callable, n_games: int = 400, seed: int = 0) -> Dict[str, Any]:
	wins = draws = losses = 0
	draw_count = 0
	turns = []
	prestige = []
	rng = np.random.RandomState(seed)
	illegal_rate = 0.0
	checks = 0

	for g in range(n_games):
		env = make_env()
		obs, info = env.reset(seed=int(rng.randint(1e9)))
		term = False
		while True:
			mask = info["action_mask"]
			checks += 1
			illegal_rate += 0 if mask.sum() > 0 else 0  # placeholder: illegal selection prevented by policy
			a = model_policy(obs, info)
			if mask[a] == 0:
				# should never happen with greedy/masked
				illegal_rate += 1
			obs, r, term, trunc, info = env.step(a)
			if term or trunc:
				if r > 0: wins += 1
				elif r < 0: losses += 1
				else:
					draws += 1
				break
		# Stats from underlying env
		base = env.env  # underlying SplendorEnv
		turns.append(base.state.turn_count)
		# previous player index
		prev_idx = (base.state.to_play - 1) % base.state.num_players
		prestige.append(base.state.players[prev_idx].prestige)
		env.close()

	n = n_games
	p = wins / max(1, n)
	ci = 1.96 * np.sqrt(p * (1 - p) / max(1, n))
	return {
		"n": n,
		"wins": wins,
		"losses": losses,
		"draws": draws,
		"win_rate": p,
		"win_rate_ci95": ci,
		"avg_turns": float(np.mean(turns)) if turns else 0.0,
		"avg_prestige": float(np.mean(prestige)) if prestige else 0.0,
		"illegal_action_rate": float(illegal_rate / max(1, checks)),
	} 