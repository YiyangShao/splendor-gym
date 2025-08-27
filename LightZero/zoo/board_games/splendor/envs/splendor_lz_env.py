import copy
import random
from typing import List, Any, Dict, Tuple, Optional
import sys
import os

import numpy as np
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from easydict import EasyDict

# Add the splendor_gym to the path so we can import it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..')))

from splendor_gym.engine import (
    SplendorState,
    legal_moves,
    apply_action,
    is_terminal,
    winner,
    initial_state,
)
from splendor_gym.engine.encode import TOTAL_ACTIONS, OBSERVATION_DIM, encode_observation
from splendor_gym.strategies.noble_strategy import noble_policy


class NobleStrategy:
    """Wrapper class for the noble strategy functions."""
    
    def choose_action(self, state: SplendorState) -> int:
        """Choose an action using the noble strategy policy."""
        obs = encode_observation(state)
        mask = legal_moves(state)
        info = {"action_mask": mask}
        return noble_policy(obs, info)


@ENV_REGISTRY.register('splendor')
class SplendorLightZeroEnv(BaseEnv):
    """
    LightZero-compatible Splendor environment for AlphaZero.
    
    This environment wraps the existing Splendor engine to provide the API expected by LightZero,
    particularly for AlphaZero's board-game path which requires:
    - legal_actions -> List[int] of legal action IDs
    - current_state() -> (board_state, board_state_scale) for MCTS simulation
    - reset(state_config) for resetting to arbitrary states during MCTS
    - step(action) for self-play progression
    - random_action() and bot_action() for evaluation
    """

    config = dict(
        env_id="Splendor",
        non_zero_sum=False,
        battle_mode='self_play_mode',
        battle_mode_in_simulation_env='self_play_mode',
        bot_action_type='noble_strategy',
        replay_path=None,
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        channel_last=False,
        scale=True,
        stop_value=1,
        alphazero_mcts_ctree=False,
        num_players=2,
        max_turns=100,  # Turn limit to prevent infinite games
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg=None):
        default_config = self.default_config()
        if cfg is not None:
            default_config.update(cfg)
        self._cfg = default_config
        
        self.battle_mode = self._cfg.battle_mode
        self.battle_mode_in_simulation_env = 'self_play_mode'
        self.channel_last = self._cfg.channel_last
        self.scale = self._cfg.scale
        self.num_players = self._cfg.num_players
        self.max_turns = self._cfg.max_turns
        
        # Action and observation spaces
        self.total_num_actions = TOTAL_ACTIONS  # 45 actions
        self.observation_dim = OBSERVATION_DIM  # 224-dim vector
        
        # Define DI-engine compatible spaces
        import gymnasium as gym
        self._action_space = gym.spaces.Discrete(self.total_num_actions)
        self._observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.observation_dim,), dtype=np.float32
        )
        self._reward_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Bot for evaluation
        self.bot_action_type = self._cfg.bot_action_type
        if self.bot_action_type == 'noble_strategy':
            self.noble_bot = NobleStrategy()
        
        # Game state
        self.state: Optional[SplendorState] = None
        self.current_player: int = 0
        self.next_player: int = 1
        
        # For compatibility with LightZero
        self.players = [0, 1]  # Player indices
        self.alphazero_mcts_ctree = self._cfg.alphazero_mcts_ctree
        
        self._env = self

    @property
    def legal_actions(self) -> List[int]:
        """Return list of legal action indices for the current state."""
        if self.state is None:
            return []
        mask = legal_moves(self.state)
        return [i for i, legal in enumerate(mask) if legal]

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the current state for MCTS simulation.
        
        Returns:
            Tuple of (board_state, board_state_scale):
            - board_state: Serializable representation of game state for MCTS reset
            - board_state_scale: Normalized observation vector for the neural network (224-dim)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # CRITICAL FIX: board_state must be serializable for MCTS, not SplendorState object
        # Use the encoded observation as the serializable board state
        raw_obs = encode_observation(self.state)
        board_state = raw_obs.astype(np.float32)  # Use serializable representation
        
        # board_state_scale: normalized observation for neural network
        if self.scale:
            # Normalize the observation to [0, 1] range
            board_state_scale = self._normalize_observation(raw_obs)
        else:
            board_state_scale = raw_obs.astype(np.float32)
        
        # Reshape to 3D for AlphaZero model compatibility: (1, 224, 1)
        board_state_scale = board_state_scale.reshape(1, 224, 1)
        
        return board_state, board_state_scale

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation to [0, 1] range using reasonable caps.
        Based on the roadmap: tokens/10, bank per color/7, prestige/20, costs/7, etc.
        """
        obs_float = obs.astype(np.float32)
        
        # Known structure of observation (from encode.py):
        # Bank (6) + Current player (13) + Opponent (13) + Board (156) + Nobles (30) + Decks (3) + Misc (3) = 224
        
        normalized = np.zeros_like(obs_float)
        idx = 0
        
        # Bank tokens (6): normalize by reasonable token cap of 10
        normalized[idx:idx+6] = np.clip(obs_float[idx:idx+6] / 10.0, 0, 1)
        idx += 6
        
        # Current player: tokens(6), bonuses(5), prestige(1), reserved_count(1)
        normalized[idx:idx+6] = np.clip(obs_float[idx:idx+6] / 10.0, 0, 1)  # tokens
        idx += 6
        normalized[idx:idx+5] = np.clip(obs_float[idx:idx+5] / 7.0, 0, 1)   # bonuses
        idx += 5
        normalized[idx] = np.clip(obs_float[idx] / 20.0, 0, 1)  # prestige
        idx += 1
        normalized[idx] = np.clip(obs_float[idx] / 3.0, 0, 1)   # reserved count
        idx += 1
        
        # Opponent: same structure as current player
        normalized[idx:idx+6] = np.clip(obs_float[idx:idx+6] / 10.0, 0, 1)  # tokens
        idx += 6
        normalized[idx:idx+5] = np.clip(obs_float[idx:idx+5] / 7.0, 0, 1)   # bonuses
        idx += 5
        normalized[idx] = np.clip(obs_float[idx] / 20.0, 0, 1)  # prestige
        idx += 1
        normalized[idx] = np.clip(obs_float[idx] / 3.0, 0, 1)   # reserved count
        idx += 1
        
        # Board cards (156): 12 cards × (present1, tier1, points1, color_onehot5, cost5)
        for card_idx in range(12):
            card_start = idx + card_idx * 13
            # present (binary)
            normalized[card_start] = obs_float[card_start]
            # tier (1, 2, or 3)
            normalized[card_start + 1] = obs_float[card_start + 1] / 3.0
            # points
            normalized[card_start + 2] = np.clip(obs_float[card_start + 2] / 5.0, 0, 1)
            # color onehot (already 0/1)
            normalized[card_start + 3:card_start + 8] = obs_float[card_start + 3:card_start + 8]
            # costs
            normalized[card_start + 8:card_start + 13] = np.clip(obs_float[card_start + 8:card_start + 13] / 7.0, 0, 1)
        idx += 156
        
        # Nobles (30): 5 × (present1, req5)
        for noble_idx in range(5):
            noble_start = idx + noble_idx * 6
            # present (binary)
            normalized[noble_start] = obs_float[noble_start]
            # requirements
            normalized[noble_start + 1:noble_start + 6] = np.clip(obs_float[noble_start + 1:noble_start + 6] / 5.0, 0, 1)
        idx += 30
        
        # Deck sizes (3)
        normalized[idx:idx+3] = np.clip(obs_float[idx:idx+3] / 40.0, 0, 1)  # reasonable deck size cap
        idx += 3
        
        # Misc: turn_count, to_play, round_over_flag
        normalized[idx] = np.clip(obs_float[idx] / (self.max_turns * 2), 0, 1)  # turn count
        normalized[idx + 1] = obs_float[idx + 1]  # to_play (already 0/1)
        normalized[idx + 2] = obs_float[idx + 2]  # round_over_flag (already 0/1)
        
        return normalized

    def reset(self, start_player_index: int = 0, init_state: Optional[SplendorState] = None, 
              katago_policy_init: bool = False, katago_game_state: Any = None) -> Dict[str, Any]:
        """
        Reset the environment, optionally to a specific state for MCTS simulation.
        
        Args:
            start_player_index: Starting player (0 or 1)
            init_state: If provided, reset to this specific state (for MCTS)
            katago_policy_init: Compatibility parameter (unused)
            katago_game_state: Compatibility parameter (unused)
            
        Returns:
            Initial observation dict
        """
        if init_state is not None:
            # CRITICAL FIX: Handle both SplendorState objects and serialized numpy arrays
            if isinstance(init_state, np.ndarray):
                # This is a serialized state from MCTS - create a fresh game instead
                # TODO: In the future, we could implement state reconstruction from encoded observation
                seed = random.randint(0, 2**31 - 1)
                self.state = initial_state(num_players=self.num_players, seed=seed)
                self.state.to_play = start_player_index
            else:
                # This is a SplendorState object - use it directly
                self.state = copy.deepcopy(init_state)
        else:
            # Fresh game start
            seed = random.randint(0, 2**31 - 1)
            self.state = initial_state(num_players=self.num_players, seed=seed)
            self.state.to_play = start_player_index
        
        self.current_player = self.state.to_play
        self.next_player = (self.current_player + 1) % self.num_players
        
        # Get observation and action mask
        _, scaled_obs = self.current_state()
        action_mask = np.array(legal_moves(self.state), dtype=np.int8)
        
        # Return observation in LightZero format
        # Get serializable board state from current_state method
        board_state, _ = self.current_state()
        
        if self.battle_mode == 'self_play_mode':
            obs = {
                'observation': scaled_obs,
                'action_mask': action_mask,
                'board': board_state,  # Use serializable board state
                'current_player_index': self.current_player,
                'to_play': self.current_player
            }
        else:
            obs = {
                'observation': scaled_obs,
                'action_mask': action_mask, 
                'board': board_state,  # Use serializable board state
                'current_player_index': self.current_player,
                'to_play': -1
            }
        
        return obs

    def step(self, action: int) -> BaseEnvTimestep:
        """
        Execute an action and return the new state.
        
        Args:
            action: Action index (0-44)
            
        Returns:
            BaseEnvTimestep object
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if is_terminal(self.state):
            raise RuntimeError("Game is already terminal. Call reset() for a new game.")
        
        # Check if action is legal
        mask = legal_moves(self.state)
        if not mask[action]:
            # Illegal action penalty - return current state
            _, scaled_obs = self.current_state()
            action_mask = np.array(legal_moves(self.state), dtype=np.int8)
            
            # CRITICAL FIX: Use serializable board state for illegal actions too
            board_state, _ = self.current_state()
            
            obs = {
                'observation': scaled_obs,
                'action_mask': action_mask,
                'board': board_state,  # Use serializable board state
                'current_player_index': self.current_player,
                'to_play': self.current_player if self.battle_mode == 'self_play_mode' else -1
            }
            return BaseEnvTimestep(obs, -1.0, False, {"illegal_action": True})
        
        # Apply the action
        old_player = self.state.to_play
        self.state = apply_action(self.state, action)
        
        # Update player tracking
        self.current_player = self.state.to_play
        self.next_player = (self.current_player + 1) % self.num_players
        
        # CRITICAL FIX: Enforce turn limit to prevent infinite games
        if self.state.turn_count >= self.max_turns:
            # Make a copy and set game_over to trigger termination
            from dataclasses import replace
            self.state = replace(self.state, game_over=True, turn_limit_reached=True)
        
        # Check if game is terminal
        done = is_terminal(self.state)
        reward = 0.0
        
        if done:
            w = winner(self.state)
            if w is None:
                # Draw
                reward = 0.0
            else:
                # Win/loss from perspective of the player who just moved
                reward = 1.0 if w == old_player else -1.0
        
        # Get new observation
        _, scaled_obs = self.current_state()
        action_mask = np.array(legal_moves(self.state), dtype=np.int8) if not done else np.zeros(self.total_num_actions, dtype=np.int8)
        
        # Get serializable board state from current_state method  
        board_state, _ = self.current_state()
        
        obs = {
            'observation': scaled_obs,
            'action_mask': action_mask,
            'board': board_state,  # Use serializable board state
            'current_player_index': self.current_player,
            'to_play': self.current_player if self.battle_mode == 'self_play_mode' else -1
        }
        
        info = {
            "to_play": self.state.to_play,
            "winner": winner(self.state) if done else None,
        }
        
        return BaseEnvTimestep(obs, reward, done, info)

    def get_done_reward(self) -> Tuple[bool, Optional[float]]:
        """
        Check if game is done and get reward from perspective of player 0.
        
        Returns:
            Tuple of (done, reward):
            - If player 0 wins: done=True, reward=1.0
            - If player 1 wins: done=True, reward=-1.0  
            - If draw: done=True, reward=0.0
            - If not done: done=False, reward=None
        """
        if self.state is None:
            return False, None
        
        done = is_terminal(self.state)
        if not done:
            return False, None
        
        w = winner(self.state)
        if w is None:
            return True, 0.0  # Draw
        elif w == 0:
            return True, 1.0  # Player 0 wins
        else:
            return True, -1.0  # Player 1 wins

    def get_done_winner(self) -> Tuple[bool, int]:
        """
        Check if game is done and return winner.
        
        Returns:
            Tuple of (done, winner):
            - If player 0 wins: done=True, winner=0
            - If player 1 wins: done=True, winner=1
            - If draw: done=True, winner=-1
            - If not done: done=False, winner=-1
        """
        if self.state is None:
            return False, -1
        
        done = is_terminal(self.state)
        if not done:
            return False, -1
        
        w = winner(self.state)
        if w is None:
            return True, -1  # Draw
        else:
            return True, w   # Winner index (0 or 1)

    def random_action(self) -> int:
        """Return a random legal action."""
        legal = self.legal_actions
        if not legal:
            raise ValueError("No legal actions available")
        return random.choice(legal)

    def bot_action(self) -> int:
        """Return an action chosen by the bot strategy."""
        if self.bot_action_type == 'noble_strategy':
            action = self.noble_bot.choose_action(self.state)
            return action
        else:
            # Fallback to random
            return self.random_action()

    def render(self, mode: str = 'human') -> None:
        """Render the current game state."""
        if self.state is None:
            print("Environment not initialized")
            return
        
        print(f"Turn {self.state.turn_count} - Player {self.state.to_play} to play")
        print(f"Bank: {dict(zip(['white', 'blue', 'green', 'red', 'black', 'gold'], self.state.bank))}")
        
        for i, player in enumerate(self.state.players):
            print(f"Player {i}: tokens={dict(zip(['white', 'blue', 'green', 'red', 'black', 'gold'], player.tokens))}, "
                  f"bonuses={dict(zip(['white', 'blue', 'green', 'red', 'black'], player.bonuses))}, "
                  f"prestige={player.prestige}")

    @property
    def observation_space(self):
        """Return the observation space."""
        return self._observation_space

    @property  
    def action_space(self):
        """Return the action space."""
        return self._action_space

    @property
    def reward_space(self):
        """Return the reward space."""
        return self._reward_space

    def seed(self, seed: Optional[int] = None, dynamic_seed: Optional[int] = None) -> None:
        """Set random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def close(self) -> None:
        """Clean up resources."""
        pass

    def __repr__(self) -> str:
        """String representation of the environment."""
        return f"SplendorLightZeroEnv(num_players={self.num_players}, max_turns={self.max_turns})"
