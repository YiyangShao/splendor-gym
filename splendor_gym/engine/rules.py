from __future__ import annotations

from typing import List, Optional, Dict

from .state import (
	SplendorState,
	PlayerState,
	Card,
	Noble,
	STANDARD_COLORS,
	COLOR_INDEX,
	initial_state as _initial_state,
)
from .encode import (
	TAKE3_COMBOS,
	TAKE3_OFFSET,
	TAKE3_COUNT,
	TAKE2_OFFSET,
	TAKE2_COUNT,
	BUY_VISIBLE_OFFSET,
	BUY_VISIBLE_COUNT,
	RESERVE_VISIBLE_OFFSET,
	RESERVE_VISIBLE_COUNT,
	RESERVE_BLIND_OFFSET,
	RESERVE_BLIND_COUNT,
	BUY_RESERVED_OFFSET,
	BUY_RESERVED_COUNT,
	TOTAL_ACTIONS,
)


def initial_state(num_players: int = 2, seed: int = 0) -> SplendorState:
	return _initial_state(num_players=num_players, seed=seed)


def legal_moves(state: SplendorState) -> List[int]:
	mask = [0] * TOTAL_ACTIONS
	player = state.players[state.to_play]
	bank = state.bank

	# Take-3: need at least 1 of each distinct color
	for idx, (a, b, c) in enumerate(TAKE3_COMBOS):
		if bank[a] >= 1 and bank[b] >= 1 and bank[c] >= 1:
			mask[TAKE3_OFFSET + idx] = 1

	# Take-2: need at least 4 of that color in bank
	for i in range(5):
		if bank[i] >= 4:
			mask[TAKE2_OFFSET + i] = 1

	# Buy visible
	for tier in (1, 2, 3):
		for slot in range(4):
			flat = BUY_VISIBLE_OFFSET + (tier - 1) * 4 + slot
			card = state.board[tier][slot]
			if card is not None and _can_afford(player, card):
				mask[flat] = 1

	# Reserve visible (if reserved < 3)
	if len(player.reserved) < 3:
		for tier in (1, 2, 3):
			for slot in range(4):
				flat = RESERVE_VISIBLE_OFFSET + (tier - 1) * 4 + slot
				card = state.board[tier][slot]
				if card is not None:
					mask[flat] = 1

	# Reserve blind (deck non-empty and reserved < 3)
	if len(player.reserved) < 3:
		for tier in (1, 2, 3):
			if state.decks[tier]:
				mask[RESERVE_BLIND_OFFSET + (tier - 1)] = 1

	# Buy reserved
	for i in range(min(3, len(player.reserved))):
		if _can_afford(player, player.reserved[i]):
			mask[BUY_RESERVED_OFFSET + i] = 1

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
	# choose exactly one if multiple available
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


def auto_return_tokens(player: PlayerState, k: int, state: SplendorState) -> Dict[str, int]:
	"""
	Return exactly k tokens. Greedy: drop colors with largest counts first; drop gold last.
	Returns a dict color->count removed.
	"""
	removed: Dict[str, int] = {c: 0 for c in STANDARD_COLORS + ["gold"]}
	if k <= 0:
		return removed
	# Indices order: prioritize non-gold from highest to lowest counts, gold last
	counts = [(i, player.tokens[i]) for i in range(5)]
	counts.sort(key=lambda x: x[1], reverse=True)
	indices = [i for i, _ in counts] + [COLOR_INDEX["gold"]]
	remaining = k
	for i in indices:
		if remaining == 0:
			break
		give = min(remaining, player.tokens[i])
		player.tokens[i] -= give
		state.bank[i] += give
		removed[STANDARD_COLORS[i] if i < 5 else "gold"] = give
		remaining -= give
	return removed


def _enforce_token_limit(player: PlayerState, state: SplendorState) -> None:
	total = sum(player.tokens)
	if total <= 10:
		return
	over = total - 10
	auto_return_tokens(player, over, state)


def apply_action(state: SplendorState, action: int) -> SplendorState:
	next_state = state.copy()
	player = next_state.players[next_state.to_play]
	bank = next_state.bank

	if TAKE3_OFFSET <= action < TAKE3_OFFSET + TAKE3_COUNT:
		# Take-3
		idx = action - TAKE3_OFFSET
		a, b, c = TAKE3_COMBOS[idx]
		bank[a] -= 1
		bank[b] -= 1
		bank[c] -= 1
		player.tokens[a] += 1
		player.tokens[b] += 1
		player.tokens[c] += 1
	elif TAKE2_OFFSET <= action < TAKE2_OFFSET + TAKE2_COUNT:
		# Take-2
		color_index = action - TAKE2_OFFSET
		bank[color_index] -= 2
		player.tokens[color_index] += 2
	elif BUY_VISIBLE_OFFSET <= action < BUY_VISIBLE_OFFSET + BUY_VISIBLE_COUNT:
		# Buy visible
		offset = action - BUY_VISIBLE_OFFSET
		tier = 1 + offset // 4
		slot = offset % 4
		card = next_state.board[tier][slot]
		assert card is not None
		_pay_for_card(player, bank, card)
		next_state.board[tier][slot] = None
		_refill_slot(next_state, tier, slot)
	elif RESERVE_VISIBLE_OFFSET <= action < RESERVE_VISIBLE_OFFSET + RESERVE_VISIBLE_COUNT:
		# Reserve visible
		offset = action - RESERVE_VISIBLE_OFFSET
		tier = 1 + offset // 4
		slot = offset % 4
		card = next_state.board[tier][slot]
		assert card is not None
		next_state.board[tier][slot] = None
		player.reserved.append(card)
		# Take gold if available
		if bank[COLOR_INDEX["gold"]] > 0:
			bank[COLOR_INDEX["gold"]] -= 1
			player.tokens[COLOR_INDEX["gold"]] += 1
		_refill_slot(next_state, tier, slot)
	elif RESERVE_BLIND_OFFSET <= action < RESERVE_BLIND_OFFSET + RESERVE_BLIND_COUNT:
		# Reserve blind
		tier = 1 + (action - RESERVE_BLIND_OFFSET)
		card = next_state.decks[tier].pop()
		player.reserved.append(card)
		if bank[COLOR_INDEX["gold"]] > 0:
			bank[COLOR_INDEX["gold"]] -= 1
			player.tokens[COLOR_INDEX["gold"]] += 1
	elif BUY_RESERVED_OFFSET <= action < BUY_RESERVED_OFFSET + BUY_RESERVED_COUNT:
		# Buy reserved
		idx = action - BUY_RESERVED_OFFSET
		card = player.reserved.pop(idx)
		_pay_for_card(player, bank, card)
	else:
		raise ValueError("Invalid action index")

	# End of turn procedures
	_grant_noble_if_applicable(player, next_state.nobles)
	_enforce_token_limit(player, next_state)

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
	best = []
	for idx, p in enumerate(state.players):
		stats = (
			p.prestige,
			- sum(p.bonuses),  # fewer purchased cards is better
			- len(p.reserved),
		)
		best.append((stats, idx))
	best.sort(reverse=True)
	if len(best) >= 2 and best[0][0] == best[1][0]:
		return None
	return best[0][1]


def is_terminal(state: SplendorState) -> bool:
	# Terminal once the round is completed (wrapped to player 0) after game_over triggered.
	return state.game_over and state.to_play == 0


def winner(state: SplendorState) -> Optional[int]:
	return state.winner_index 