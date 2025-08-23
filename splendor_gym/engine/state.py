from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random

# Token colors: 5 standard + 1 gold (wild)
TOKEN_COLORS = ["white", "blue", "green", "red", "black", "gold"]
STANDARD_COLORS = TOKEN_COLORS[:-1]
COLOR_INDEX = {c: i for i, c in enumerate(TOKEN_COLORS)}

# Splendor defaults for 2 players
DEFAULT_BANK = {
	"white": 4,
	"blue": 4,
	"green": 4,
	"red": 4,
	"black": 4,
	"gold": 5,
}


@dataclass
class Card:
	id: int
	tier: int
	color: str  # one of STANDARD_COLORS
	points: int
	cost: Dict[str, int]


@dataclass
class Noble:
	id: int
	requirements: Dict[str, int]
	points: int = 3


@dataclass
class PlayerState:
	tokens: List[int] = field(default_factory=lambda: [0] * len(TOKEN_COLORS))
	bonuses: List[int] = field(default_factory=lambda: [0] * len(STANDARD_COLORS))
	prestige: int = 0
	reserved: List[Card] = field(default_factory=list)
	nobles: List[Noble] = field(default_factory=list)

	def can_afford(self, card: Card) -> Tuple[bool, List[int]]:
		cost_remaining = []
		gold_needed = 0
		for i, color in enumerate(STANDARD_COLORS):
			req = card.cost.get(color, 0)
			discounted = max(0, req - self.bonuses[i])
			pay_with_color = min(self.tokens[i], discounted)
			remaining = discounted - pay_with_color
			cost_remaining.append(discounted)
			gold_needed += remaining
		return (self.tokens[COLOR_INDEX["gold"]] >= gold_needed, cost_remaining)


@dataclass
class SplendorState:
	num_players: int
	bank: List[int]  # tokens by TOKEN_COLORS order
	players: List[PlayerState]
	board: Dict[int, List[Optional[Card]]]  # tier -> 4 visible slots (None if empty)
	decks: Dict[int, List[Card]]  # tier -> remaining deck top at end
	nobles: List[Optional[Noble]]  # visible nobles (None if taken)
	to_play: int = 0
	turn_count: int = 0
	game_over: bool = False
	winner_index: Optional[int] = None

	def copy(self) -> "SplendorState":
		# shallow copy is sufficient for our immutable Card/Noble objects, deep-copy lists
		return SplendorState(
			num_players=self.num_players,
			bank=list(self.bank),
			players=[PlayerState(tokens=list(p.tokens), bonuses=list(p.bonuses), prestige=p.prestige, reserved=list(p.reserved), nobles=list(p.nobles)) for p in self.players],
			board={tier: list(slots) for tier, slots in self.board.items()},
			decks={tier: list(deck) for tier, deck in self.decks.items()},
			nobles=list(self.nobles),
			to_play=self.to_play,
			turn_count=self.turn_count,
			game_over=self.game_over,
			winner_index=self.winner_index,
		)


# Minimal bootstrap card/noble factories (to be replaced by JSON loading in rules)
def generate_mock_decks(seed: int = 0) -> Tuple[Dict[int, List[Card]], Dict[int, List[Optional[Card]]]]:
	random.seed(seed)
	card_id = 0
	decks: Dict[int, List[Card]] = {1: [], 2: [], 3: []}
	board: Dict[int, List[Optional[Card]]] = {1: [None] * 4, 2: [None] * 4, 3: [None] * 4}
	for tier in (1, 2, 3):
		for _ in range(20):
			color = random.choice(STANDARD_COLORS)
			points = 0 if tier == 1 else (1 if tier == 2 else random.choice([2, 3, 4]))
			cost = {c: random.randint(0, 4 if tier == 1 else 6) for c in STANDARD_COLORS}
			cost[color] = max(0, cost[color] - (1 if tier == 1 else 2))
			decks[tier].append(Card(id=card_id, tier=tier, color=color, points=points, cost=cost))
			card_id += 1
		random.shuffle(decks[tier])
		for i in range(4):
			board[tier][i] = decks[tier].pop() if decks[tier] else None
	return decks, board


def generate_mock_nobles(num: int = 3, start_id: int = 1000) -> List[Optional[Noble]]:
	nobles: List[Optional[Noble]] = []
	for i in range(num):
		req = {c: 3 for c in random.sample(STANDARD_COLORS, 3)}
		nobles.append(Noble(id=start_id + i, requirements=req))
	return nobles


def initial_state(num_players: int = 2, seed: int = 0) -> SplendorState:
	decks, board = generate_mock_decks(seed)
	nobles = generate_mock_nobles(3)
	bank = [DEFAULT_BANK[c] for c in TOKEN_COLORS]
	players = [PlayerState() for _ in range(num_players)]
	return SplendorState(
		num_players=num_players,
		bank=bank,
		players=players,
		board=board,
		decks=decks,
		nobles=nobles,
		to_play=0,
		turn_count=0,
		game_over=False,
		winner_index=None,
	) 