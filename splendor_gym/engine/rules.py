from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import itertools

from .state import (
	SplendorState,
	PlayerState,
	Card,
	Noble,
	TOKEN_COLORS,
	STANDARD_COLORS,
	COLOR_INDEX,
	initial_state as _initial_state,
)

# Action encoding
# We map actions to a flat index:
# 0..9   : Take-3 (combinations of 3 distinct standard colors)
# 10..14 : Take-2 (double from one standard color)
# 15..26 : Reserve visible cards (12 visible: 3 tiers x 4 slots)
# 27..29 : Reserve blind from deck tier 1..3
# 30..41 : Buy visible cards (12 visible)
# 42..44 : Buy from reserved slots 0..2
# Total 45 actions

TAKE3_COUNT = 10
TAKE2_COUNT = 5
RESERVE_VISIBLE_COUNT = 12
RESERVE_BLIND_COUNT = 3
BUY_VISIBLE_COUNT = 12
BUY_RESERVED_COUNT = 3

TOTAL_ACTIONS = (
	TAKE3_COUNT
	+ TAKE2_COUNT
	+ RESERVE_VISIBLE_COUNT
	+ RESERVE_BLIND_COUNT
	+ BUY_VISIBLE_COUNT
	+ BUY_RESERVED_COUNT
)

# Precompute combinations of 3 colors for TAKE3
TAKE3_COMBOS: List[Tuple[int, int, int]] = list(itertools.combinations(range(5), 3))


def initial_state(num_players: int = 2, seed: int = 0) -> SplendorState:
	return _initial_state(num_players=num_players, seed=seed)


def legal_moves(state: SplendorState) -> List[int]:
	mask = [0] * TOTAL_ACTIONS
	player = state.players[state.to_play]
	bank = state.bank

	# Take-3: need at least 1 of each color
	for idx, (a, b, c) in enumerate(TAKE3_COMBOS):
		if bank[a] >= 1 and bank[b] >= 1 and bank[c] >= 1:
			mask[idx] = 1

	# Take-2: need at least 4 of that color in bank
	for i in range(5):
		if bank[i] >= 4:
			mask[TAKE3_COUNT + i] = 1

	# Reserve visible
	base = TAKE3_COUNT + TAKE2_COUNT
	for tier in (1, 2, 3):
		for slot in range(4):
			flat = base + (tier - 1) * 4 + slot
			card = state.board[tier][slot]
			if card is not None and len(player.reserved) < 3:
				mask[flat] = 1

	# Reserve blind
	base += RESERVE_VISIBLE_COUNT
	for tier in (1, 2, 3):
		flat = base + (tier - 1)
		if state.decks[tier] and len(player.reserved) < 3:
			mask[flat] = 1

	# Buy visible
	base += RESERVE_BLIND_COUNT
	for tier in (1, 2, 3):
		for slot in range(4):
			flat = base + (tier - 1) * 4 + slot
			card = state.board[tier][slot]
			if card is not None and _can_afford(player, card):
				mask[flat] = 1

	# Buy reserved
	base += BUY_VISIBLE_COUNT
	for i in range(3):
		flat = base + i
		if i < len(player.reserved) and _can_afford(player, player.reserved[i]):
			mask[flat] = 1

	return mask


def _can_afford(player: PlayerState, card: Card) -> bool:
	can, _ = player.can_afford(card)
	return can


def _pay_for_card(player: PlayerState, bank: List[int], card: Card) -> None:
	# Determine payment including gold
	gold_available = player.tokens[COLOR_INDEX["gold"]]
	gold_spent = 0
	for i, color in enumerate(STANDARD_COLORS):
		req = card.cost.get(color, 0)
		discounted = max(0, req - player.bonuses[i])
		spend_color = min(player.tokens[i], discounted)
		player.tokens[i] -= spend_color
		bank[i] += spend_color
		remaining = discounted - spend_color
		if remaining > 0:
			use_gold = min(remaining, gold_available - gold_spent)
			gold_spent += use_gold
			# gold returns to bank as gold
	player.tokens[COLOR_INDEX["gold"]] -= gold_spent
	bank[COLOR_INDEX["gold"]] += gold_spent

	# Gain bonus and points
	player.bonuses[STANDARD_COLORS.index(card.color)] += 1
	player.prestige += card.points


def _refill_slot(state: SplendorState, tier: int, slot: int) -> None:
	if state.decks[tier]:
		state.board[tier][slot] = state.decks[tier].pop()
	else:
		state.board[tier][slot] = None


def _grant_noble_if_applicable(player: PlayerState, nobles: List[Optional[Noble]]) -> None:
	for idx, noble in enumerate(nobles):
		if noble is None:
			continue
		meets = True
		for i, color in enumerate(STANDARD_COLORS):
			req = noble.requirements.get(color, 0)
			if player.bonuses[i] < req:
				meets = False
				break
		if meets:
			player.nobles.append(noble)
			player.prestige += noble.points
			nobles[idx] = None
			break


def _enforce_token_limit(player: PlayerState, bank: List[int]) -> None:
	total = sum(player.tokens)
	if total <= 10:
		return
		# Simple heuristic discard: return gold first, then highest color counts
	over = total - 10
	for i in [COLOR_INDEX["gold"], 0, 1, 2, 3, 4]:
		if over == 0:
			break
		give = min(over, player.tokens[i])
		player.tokens[i] -= give
		bank[i] += give
		over -= give


def apply_action(state: SplendorState, action: int) -> SplendorState:
	next_state = state.copy()
	player = next_state.players[next_state.to_play]
	bank = next_state.bank

	if action < TAKE3_COUNT:
		# Take-3
		a, b, c = TAKE3_COMBOS[action]
		bank[a] -= 1
		bank[b] -= 1
		bank[c] -= 1
		player.tokens[a] += 1
		player.tokens[b] += 1
		player.tokens[c] += 1
	elif action < TAKE3_COUNT + TAKE2_COUNT:
		# Take-2
		color_index = action - TAKE3_COUNT
		bank[color_index] -= 2
		player.tokens[color_index] += 2
	elif action < TAKE3_COUNT + TAKE2_COUNT + RESERVE_VISIBLE_COUNT:
		# Reserve visible
		offset = action - (TAKE3_COUNT + TAKE2_COUNT)
		ier = 1 + offset // 4
		slot = offset % 4
		card = next_state.board[ier][slot]
		assert card is not None
		next_state.board[ier][slot] = None
		player.reserved.append(card)
		# Take gold if available
		if bank[COLOR_INDEX["gold"]] > 0:
			bank[COLOR_INDEX["gold"]] -= 1
			player.tokens[COLOR_INDEX["gold"]] += 1
		_refill_slot(next_state, ier, slot)
	elif action < TAKE3_COUNT + TAKE2_COUNT + RESERVE_VISIBLE_COUNT + RESERVE_BLIND_COUNT:
		# Reserve blind
		tier = 1 + (action - (TAKE3_COUNT + TAKE2_COUNT + RESERVE_VISIBLE_COUNT))
		card = next_state.decks[tier].pop()
		player.reserved.append(card)
		if bank[COLOR_INDEX["gold"]] > 0:
			bank[COLOR_INDEX["gold"]] -= 1
			player.tokens[COLOR_INDEX["gold"]] += 1
	elif action < TAKE3_COUNT + TAKE2_COUNT + RESERVE_VISIBLE_COUNT + RESERVE_BLIND_COUNT + BUY_VISIBLE_COUNT:
		# Buy visible
		offset = action - (TAKE3_COUNT + TAKE2_COUNT + RESERVE_VISIBLE_COUNT + RESERVE_BLIND_COUNT)
		ier = 1 + offset // 4
		slot = offset % 4
		card = next_state.board[ier][slot]
		assert card is not None
		_pay_for_card(player, bank, card)
		next_state.board[ier][slot] = None
		_refill_slot(next_state, ier, slot)
	else:
		# Buy reserved
		idx = action - (
			TAKE3_COUNT
			+ TAKE2_COUNT
			+ RESERVE_VISIBLE_COUNT
			+ RESERVE_BLIND_COUNT
			+ BUY_VISIBLE_COUNT
		)
		card = player.reserved.pop(idx)
		_pay_for_card(player, bank, card)

	# End of turn procedures
	_grant_noble_if_applicable(player, next_state.nobles)
	_enforce_token_limit(player, bank)

	# Check end condition (15+ points). End round after all players play once.
	if player.prestige >= 15:
		next_state.game_over = True

	# Advance turn
	next_state.turn_count += 1
	next_state.to_play = (next_state.to_play + 1) % next_state.num_players

	# If end triggered and wrapped to player 0, pick winner
	if next_state.game_over and next_state.to_play == 0:
		_winner_index = compute_winner(next_state)
		next_state.winner_index = _winner_index
		return next_state

	return next_state


def compute_winner(state: SplendorState) -> Optional[int]:
	# Highest prestige, tie-breaker: fewer cards, then fewer reserved
	best: List[int] = []
	for idx, p in enumerate(state.players):
		stats = (
			p.prestige,
			- sum(p.bonuses),  # more cards means worse due to tie-breaker fewer is better
			- len(p.reserved),
		)
		best.append((stats, idx))
	best.sort(reverse=True)
	if len(best) >= 2 and best[0][0] == best[1][0]:
		return None
	return best[0][1]


def is_terminal(state: SplendorState) -> bool:
	# Terminal once the round is completed (wrapped to player 0) after game_over triggered.
	# Covers wins and ties.
	return state.game_over and state.to_play == 0


def winner(state: SplendorState) -> Optional[int]:
	return state.winner_index 