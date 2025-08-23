from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

from ..engine import (
	SplendorState,
	PlayerState,
	legal_moves,
	apply_action,
	is_terminal,
	winner,
	initial_state,
)
from ..engine.state import TOKEN_COLORS, STANDARD_COLORS
from ..engine.rules import TOTAL_ACTIONS


class SplendorEnv(gym.Env):
	metadata = {"render.modes": ["human", "ansi"], "name": "Splendor-v0"}

	def __init__(self, num_players: int = 2, render_mode: str | None = None, seed: int | None = None):
		super().__init__()
		self.num_players = num_players
		self.render_mode = render_mode
		self._rng = np.random.default_rng(seed)

		self.action_space = spaces.Discrete(TOTAL_ACTIONS)
		self.observation_space = spaces.Box(
			low=0, high=30, shape=(self._obs_dim(),), dtype=np.int32
		)

		self.state: SplendorState | None = None
		self.current_player: int = 0

	def _obs_dim(self) -> int:
		# bank 6 + current player tokens 6 + bonuses 5 + prestige 1 + opp tokens 6 + opp bonuses 5 + opp prestige 1
		# board: 12 cards x (tier 1 + points 1 + color 1 + cost5) = 12 x 8 = 96
		# nobles: 3 x (requirements5 + present1) = 18
		# turn count 1 + to_play 1
		return 6 + 6 + 5 + 1 + 6 + 5 + 1 + 96 + 18 + 2

	def _encode(self, state: SplendorState) -> np.ndarray:
		vec = []
		# bank
		vec.extend(state.bank)
		# current player and a single opponent summary (2p assumption)
		p = state.players[state.to_play]
		op = state.players[1 - state.to_play] if state.num_players == 2 else state.players[(state.to_play + 1) % state.num_players]
		vec.extend(p.tokens)
		vec.extend(p.bonuses)
		vec.append(p.prestige)
		vec.extend(op.tokens)
		vec.extend(op.bonuses)
		vec.append(op.prestige)
		# board 12 cards
		for tier in (1, 2, 3):
			for slot in range(4):
				card = state.board[tier][slot]
				if card is None:
					vec.extend([0, 0, 0] + [0] * 5)
				else:
					vec.append(tier)
					vec.append(card.points)
					vec.append(STANDARD_COLORS.index(card.color) + 1)
					for c in STANDARD_COLORS:
						vec.append(card.cost.get(c, 0))
		# nobles 3
		for i in range(3):
			noble = state.nobles[i] if i < len(state.nobles) else None
			if noble is None:
				vec.extend([0, 0, 0, 0, 0, 0])
			else:
				for c in STANDARD_COLORS:
					vec.append(noble.requirements.get(c, 0))
				vec.append(1)
		# misc
		vec.append(state.turn_count)
		vec.append(state.to_play)
		return np.array(vec, dtype=np.int32)

	def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
		if seed is not None:
			self._rng = np.random.default_rng(seed)
		self.state = initial_state(num_players=self.num_players, seed=int(self._rng.integers(0, 1e9)))
		self.current_player = self.state.to_play
		obs = self._encode(self.state)
		mask = np.array(legal_moves(self.state), dtype=np.int8)
		info = {"action_mask": mask, "to_play": self.state.to_play}
		return obs, info

	def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
		assert self.state is not None, "Call reset() first"
		mask = legal_moves(self.state)
		if not (0 <= action < self.action_space.n and mask[action] == 1):
			# Illegal action: treat as no-op with small penalty to discourage
			reward = -0.01
			terminated = False
			truncated = False
			info = {"illegal_action": True, "action_mask": np.array(mask, dtype=np.int8), "to_play": self.state.to_play}
			return self._encode(self.state), reward, terminated, truncated, info

		self.state = apply_action(self.state, action)
		obs = self._encode(self.state)
		terminated = is_terminal(self.state)
		reward = 0.0
		if terminated:
			w = winner(self.state)
			if w is None:
				reward = 0.0
			elif w == self.current_player:
				reward = 1.0
			else:
				reward = -1.0
		truncated = False
		mask = np.array(legal_moves(self.state), dtype=np.int8) if not terminated else np.zeros(self.action_space.n, dtype=np.int8)
		info = {"action_mask": mask, "to_play": self.state.to_play}
		return obs, float(reward), bool(terminated), bool(truncated), info

	def render(self):
		if self.render_mode not in ("human", "ansi"):
			return
		assert self.state is not None
		p = self.state.players[self.state.to_play]
		print(f"Turn {self.state.turn_count} — Player {self.state.to_play}")
		print(f"Bank: {dict(zip(TOKEN_COLORS, self.state.bank))}")
		print(f"You: tokens={dict(zip(TOKEN_COLORS, p.tokens))} bonuses={dict(zip(STANDARD_COLORS, p.bonuses))} pp={p.prestige}")


def make(num_players: int = 2, render_mode: str | None = None, seed: int | None = None) -> SplendorEnv:
	return SplendorEnv(num_players=num_players, render_mode=render_mode, seed=seed) 