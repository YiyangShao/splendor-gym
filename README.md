# Splendor Gymnasium Environment

A Gymnasium-compatible environment for the board game Splendor, designed for RL research (PPO, AlphaZero, MuZero, Decision Transformer) and compatibility with frameworks (SB3, CleanRL, LightZero).

## Quick start

```bash
# Install (editable)
pip install -e .[dev]

# Run tests
pytest -q

# Smoke test
python -m splendor_gym.scripts.random_rollout --episodes 3
```

## Project structure

```
splendor_gym/
  engine/
    __init__.py
    state.py
    rules.py
    cards.json
    nobles.json
  envs/
    __init__.py
    splendor_env.py
  tests/
    test_rules.py
    test_env.py
  docs/
    splendor_env.md
  scripts/
    random_rollout.py
```

## Status
- Phase 1: Engine — in progress
- Phase 2: Gym Env — in progress
- Phase 3: Debug & Validation — TBA
- Phase 4: RL Experiments — TBA

See `docs/splendor_env.md` for the design and API details. 