"""
Enhanced Splendor Environment with better observation encoding and reward shaping.
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

from splendor_gym.engine import (
    SplendorState, legal_moves, apply_action, is_terminal, winner, initial_state,
)
from splendor_gym.engine.state import TOKEN_COLORS, STANDARD_COLORS, PlayerState
from splendor_gym.engine.encode import TOTAL_ACTIONS


def enhanced_encode_observation(state: SplendorState) -> np.ndarray:
    """Enhanced observation encoding with affordability info and strategic features."""
    vec = []
    
    # Basic state info (bank, players, board, nobles, decks, misc) 
    # - same as original but enhanced with strategic info
    
    # 1. Bank tokens (6)
    vec.extend(state.bank)
    
    # 2. Current player detailed info (13)
    p = state.players[state.to_play]
    vec.extend(p.tokens)  # 6
    vec.extend(p.bonuses)  # 5  
    vec.append(p.prestige)  # 1
    vec.append(len(p.reserved))  # 1
    
    # 3. Opponent detailed info (13) 
    opp = state.players[(state.to_play + 1) % state.num_players]
    vec.extend(opp.tokens)  # 6
    vec.extend(opp.bonuses)  # 5
    vec.append(opp.prestige)  # 1
    vec.append(len(opp.reserved))  # 1
    
    # 4. **NEW: Strategic features for current player (8)**
    # Token efficiency
    total_tokens = sum(p.tokens)
    vec.append(min(total_tokens, 15))  # Capped at 15 for encoding
    vec.append(1 if total_tokens >= 9 else 0)  # Close to limit warning
    vec.append(1 if total_tokens >= 10 else 0)  # At limit warning
    
    # Noble proximity - how close to each visible noble  
    noble_distances = []
    for noble in state.nobles:  # Only 3 nobles in 2-player game
        if noble is None:
            noble_distances.append(10)  # Max distance
        else:
            # Calculate how many more bonuses needed for this noble
            distance = 0
            for color in STANDARD_COLORS:
                required = noble.requirements.get(color, 0)
                have = p.bonuses[STANDARD_COLORS.index(color)]
                distance += max(0, required - have)
            noble_distances.append(min(distance, 10))  # Cap at 10
    
    vec.extend(noble_distances)  # 3 nobles
    
    # Opponent threat level - how close opponent is to winning
    opp_threat = min(opp.prestige, 20)  # Cap at 20
    vec.append(opp_threat)  # 1
    vec.append(1 if opp.prestige >= 12 else 0)  # Opponent close to win
    
    # 5. Board with affordability info (12 cards × 14 = 168)
    for tier in (1, 2, 3):
        for slot in range(4):
            card = state.board[tier][slot]
            if card is None:
                vec.extend([0] * 14)  # Empty slot
            else:
                vec.append(1)  # Present
                vec.append(tier)  # Tier
                vec.append(card.points)  # Points
                
                # Color one-hot (5)
                onehot = [0] * 5
                onehot[STANDARD_COLORS.index(card.color)] = 1
                vec.extend(onehot)
                
                # Cost (5)
                for color in STANDARD_COLORS:
                    vec.append(card.cost.get(color, 0))
                
                # **NEW: Can afford this card? (1)**
                can_afford, _ = p.can_afford(card)
                vec.append(1 if can_afford else 0)
    
    # 6. Nobles with proximity info (3 × 7 = 21) - only 3 nobles in 2-player game
    for i in range(len(state.nobles)):
        noble = state.nobles[i]
        if noble is not None:
            vec.append(1)  # Present
            # Requirements (5)
            for color in STANDARD_COLORS:
                vec.append(noble.requirements.get(color, 0))
            # **NEW: Distance to noble (1)**
            distance = noble_distances[i] if i < len(noble_distances) else 10
            vec.append(min(distance, 10))
        else:
            vec.extend([0] * 7)  # Empty noble slot
    
    # 7. Deck sizes (3)
    for tier in (1, 2, 3):
        vec.append(len(state.decks[tier]))
    
    # 8. Game state (4)
    vec.append(state.turn_count)
    vec.append(state.to_play)
    vec.append(state.move_count)
    vec.append(1 if state.game_over else 0)
    
    return np.array(vec, dtype=np.int32)


def calculate_shaped_reward(
    old_state: SplendorState, 
    new_state: SplendorState, 
    action: int,
    current_player: int,
    terminated: bool
) -> float:
    """Calculate shaped reward with intermediate bonuses."""
    reward = 0.0
    
    if terminated:
        # Terminal rewards
        w = winner(new_state)
        if w is None and getattr(new_state, "turn_limit_reached", False):
            reward = -0.2  # Draw penalty
        else:
            reward = 0.0 if w is None else (2.0 if w == current_player else -2.0)
    else:
        # Intermediate rewards
        old_player = old_state.players[current_player]
        new_player = new_state.players[current_player]
        
        # Prestige progress bonus
        prestige_gain = new_player.prestige - old_player.prestige
        reward += prestige_gain * 0.1  # +0.1 per prestige point
        
        # Noble bonus
        noble_gain = len(new_player.nobles) - len(old_player.nobles) 
        reward += noble_gain * 0.5  # +0.5 per noble acquired
        
        # Efficiency penalty for hoarding tokens
        total_tokens = sum(new_player.tokens)
        if total_tokens > 10:
            reward -= 0.1  # Penalty for exceeding token limit
        elif total_tokens >= 9:
            reward -= 0.02  # Small penalty for approaching limit
            
        # Bonus acquisition reward
        bonus_gain = sum(new_player.bonuses) - sum(old_player.bonuses)
        reward += bonus_gain * 0.05  # +0.05 per bonus card
    
    return reward


class EnhancedSplendorEnv(gym.Env):
    """Enhanced Splendor environment with better observation encoding and reward shaping."""
    
    metadata = {"render_modes": ["human"], "name": "EnhancedSplendor-v0"}
    
    def __init__(self, num_players: int = 2, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        if num_players != 2:
            raise NotImplementedError("Current env supports 2 players only.")
        self.num_players = num_players
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        # Enhanced observation space - calculate new dimension
        enhanced_obs_dim = self._calculate_obs_dim()
        self.observation_space = spaces.Box(
            low=0, high=50, shape=(enhanced_obs_dim,), dtype=np.int32
        )
        
        self.state: SplendorState | None = None
        self.current_player: int = 0
    
    def _calculate_obs_dim(self) -> int:
        """Calculate enhanced observation dimension."""
        return (
            6 +      # Bank
            13 +     # Current player  
            13 +     # Opponent
            8 +      # Strategic features (3 token + 3 noble + 2 opponent)
            168 +    # Board with affordability (12 × 14)
            21 +     # Nobles with proximity (3 × 7) - 2-player game has 3 nobles
            3 +      # Deck sizes
            4        # Game state
        )  # Total: 236 dimensions
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.state = initial_state(num_players=self.num_players, seed=engine_seed)
        self.current_player = self.state.to_play
        
        obs = enhanced_encode_observation(self.state)
        mask = np.array(legal_moves(self.state), dtype=np.int8)
        info = {"action_mask": mask, "to_play": self.state.to_play}
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if is_terminal(self.state):
            raise RuntimeError("Cannot call step() after episode termination. Call reset().")
        
        mask = legal_moves(self.state)
        if np.sum(mask) == 0:
            self.state.game_over = True
            self.state.winner_index = None
            obs = enhanced_encode_observation(self.state)
            return obs, 0.0, True, False, {
                "action_mask": np.zeros(self.action_space.n, dtype=np.int8),
                "to_play": self.state.to_play, 
                "draw": True
            }
        
        if not (0 <= action < self.action_space.n):
            raise ValueError("Action out of bounds for action_space")
        if mask[action] != 1:
            obs = enhanced_encode_observation(self.state)
            return obs, -0.05, False, False, {  # Increased illegal action penalty
                "illegal_action": True,
                "action_mask": np.array(mask, dtype=np.int8),
                "to_play": self.state.to_play
            }
        
        # Store old state for reward shaping
        old_state = self.state.copy()
        self.state = apply_action(self.state, action)
        
        obs = enhanced_encode_observation(self.state)
        terminated = is_terminal(self.state)
        
        # Calculate shaped reward
        reward = calculate_shaped_reward(
            old_state, self.state, action, self.current_player, terminated
        )
        
        mask = np.array(legal_moves(self.state), dtype=np.int8) if not terminated else np.zeros(self.action_space.n, dtype=np.int8)
        info = {"action_mask": mask, "to_play": self.state.to_play}
        if terminated and getattr(self.state, "turn_limit_reached", False):
            info["turn_limit"] = True
            
        return obs, float(reward), bool(terminated), False, info

# Calculate the actual observation dimension
ENHANCED_OBS_DIM = 236
