import gymnasium as gym
import numpy as np


class SelfPlayWrapper(gym.Wrapper):
	"""
	Expose a single-agent MDP by:
	- Returning obs/action_mask for the current player only
	- Internally playing the opponent's move whenever to_play == 1
	- Returning reward from the current player's POV (+1 win, -1 loss, 0 otherwise)
	"""

	def __init__(self, env: gym.Env, opponent_policy, random_starts: bool = True, opponent_supplier=None):
		super().__init__(env)
		self.opponent_policy = opponent_policy
		self.random_starts = random_starts
		self.opponent_supplier = opponent_supplier
		self._opp_policy = opponent_policy

	def reset(self, **kwargs):
		# Sample opponent policy for this episode if supplier provided
		if self.opponent_supplier is not None:
			self._opp_policy = self.opponent_supplier()
		else:
			self._opp_policy = self.opponent_policy

		obs, info = self.env.reset(**kwargs)
		# Optional random starts: if opponent to play and coin flip, let opponent move once
		if self.random_starts and info.get("to_play", 0) == 1 and np.random.rand() < 0.5:
			a = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(a)
			if term or trunc:
				return obs, info
		# Advance until it's our turn (to_play == 0) or terminal
		while info.get("to_play", 0) == 1:
			a = self._opp_policy(obs, info)
			obs, _, term, trunc, info = self.env.step(a)
			if term or trunc:
				break
		return obs, info

	def step(self, action):
		# Our move (we are always player 0)
		obs, reward, term, trunc, info = self.env.step(action)
		if term or trunc:
			# Reward is from player 0 POV: +1 if we win, -1 if we lose, 0 if draw
			return obs, reward, term, trunc, info
		
		# Opponent move (should be player 1's turn)
		if info.get("to_play", 0) == 1:
			a = self._opp_policy(obs, info)
			obs, opp_reward, term, trunc, info = self.env.step(a)
			# Reward is from current player's perspective (who just moved)
			# If game ends, we need to flip the reward since it's from opponent's perspective
			if term or trunc:
				# Opponent just moved and got their reward, we need the opposite
				reward = -opp_reward  # Flip the reward for our perspective
			else:
				reward = 0.0
			return obs, reward, term, trunc, info
		
		# This should never happen - if not terminal and not opponent's turn, something is wrong
		raise RuntimeError(f"Invalid state: game not terminal but to_play={info.get('to_play', 'unknown')} (expected 1 for opponent)")


def random_opponent(obs, info):
	mask = info.get("action_mask")
	if mask is None:
		return 0
	legal = np.flatnonzero(mask)
	if len(legal) == 0:
		return 0
	return int(np.random.choice(legal)) 