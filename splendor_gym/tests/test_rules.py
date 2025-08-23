import numpy as np

from splendor_gym.engine import initial_state, legal_moves, apply_action, is_terminal
from splendor_gym.engine.rules import TOTAL_ACTIONS


def test_initial_legal_moves_nonempty():
	state = initial_state(seed=42)
	mask = legal_moves(state)
	assert len(mask) == TOTAL_ACTIONS
	assert any(mask), "At least one legal move at start"


def test_take3_decreases_bank():
	state = initial_state(seed=1)
	mask = legal_moves(state)
	# pick first legal action
	action = next(i for i, v in enumerate(mask) if v == 1)
	next_state = apply_action(state, action)
	if action < 10:
		# take-3
		assert sum(next_state.bank) == sum(state.bank) - 3
	assert next_state.turn_count == state.turn_count + 1


def test_terminal_flag_when_points_reached():
	state = initial_state(seed=3)
	# Force prestige to trigger end
	state.players[state.to_play].prestige = 15
	next_state = apply_action(state, 10)  # any legal action style
	# not terminal until round completion (when to_play wraps to 0)
	if next_state.to_play == 0:
		assert next_state.game_over
	else:
		# advance to wrap
		for _ in range(state.num_players - 1):
			next_state = apply_action(next_state, next(i for i, v in enumerate(legal_moves(next_state)) if v))
		assert next_state.game_over 