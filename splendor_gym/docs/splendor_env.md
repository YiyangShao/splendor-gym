# Splendor Gymnasium Environment — Design & Implementation Guide

This document tracks the design and implementation details for a Gymnasium-compatible Splendor environment intended for RL research and self-play training (AlphaZero-style).

## Goals
- RL-ready, Gymnasium-compatible
- Action masking for invalid moves
- Full ruleset: tokens, purchases, reserves, nobles, end conditions
- Deterministic seeds for reproducibility

## Engine Layer
- Pure Python rule engine, framework-agnostic
- State includes: bank (6 tokens), players, board (12 visible cards), decks, nobles, turn, player index
- API:
  - `legal_moves(state) -> mask`
  - `apply_action(state, action) -> new_state`
  - `is_terminal(state) -> bool`
  - `winner(state) -> Optional[int]`

## Actions
Flattened discrete space of size 45:
- 0..9: Take-3 distinct colors (10 combos)
- 10..14: Take-2 same color (requires bank >= 4)
- 15..26: Reserve visible (3 tiers x 4 slots)
- 27..29: Reserve blind (tier 1..3)
- 30..41: Buy visible (3 tiers x 4 slots)
- 42..44: Buy one of up to 3 reserved cards

## Observations
Fixed-size vector encoding:
- Bank (6)
- Current player: tokens (6), bonuses (5), prestige (1)
- Opponent summary: tokens (6), bonuses (5), prestige (1)
- Board: 12 cards × [tier, points, color_idx+1, cost5] = 96
- Nobles: 3 × [req5, present] = 18
- Turn count (1), to_play (1)

## Rewards
- 0 for all non-terminal steps
- +1 win / -1 loss / 0 draw at terminal

## Turn Handling
- Single-agent per step, `to_play` indicates the acting player
- `step` applies action for `to_play` and switches to next player

## Rendering
- Text render: prints bank and current player snapshot

## Roadmap
- Phase 1: Solidify rules, JSON cards/nobles
- Phase 2: Env adapter, masking (done at basic level)
- Phase 3: Validation with env checker, random rollouts
- Phase 4: Algorithms integration 