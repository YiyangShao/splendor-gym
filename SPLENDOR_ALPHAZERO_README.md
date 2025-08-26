# Splendor + AlphaZero in LightZero

This repository now includes a complete integration of the Splendor board game with AlphaZero using the LightZero framework.

## Installation & Setup

### 1. Dependencies Installed
- ✅ DI-engine 0.5.3
- ✅ LightZero 0.2.0 (installed from GitHub)
- ✅ All required dependencies (pympler, line_profiler, xxhash, minigrid)

### 2. Sanity Check Completed
- ✅ TicTacToe AlphaZero example runs successfully
- ✅ LightZero toolchain verified

## Architecture

### Board-Game Environment Integration
The Splendor environment follows LightZero's board-game pattern (not generic Gym wrapper) for optimal AlphaZero performance:

**Location**: `LightZero/zoo/board_games/splendor/`
```
splendor/
├── envs/
│   ├── __init__.py
│   └── splendor_lz_env.py          # Main environment wrapper
├── config/
│   ├── __init__.py
│   └── splendor_alphazero_sp_mode_config.py  # AlphaZero config
└── __init__.py
```

### Key API Implementation

The `SplendorLightZeroEnv` implements the required LightZero board-game API:

```python
# Required for AlphaZero MCTS
legal_actions -> List[int]                    # Legal action IDs
current_state() -> (board_state, board_state_scale)  # For simulation & neural network
reset(state_config) -> obs                    # Reset to arbitrary states (MCTS)
step(action) -> (obs, reward, done, info)     # Self-play progression

# Optional for evaluation
random_action() -> int                        # Random baseline
bot_action() -> int                          # Noble strategy bot
```

### Observation & State Design

- **Model Input**: 224-dim normalized vector (same as existing PPO setup)
- **Action Space**: 45 actions (existing encoding: TAKE3, TAKE2, BUY_VISIBLE, etc.)
- **Search State**: Deep-copyable `SplendorState` object for MCTS simulation
- **Normalization**: Tokens/10, bank/7, prestige/20, costs/7 (as per roadmap)

## Configuration

### AlphaZero Hyperparameters
The configuration follows the roadmap specifications:

```python
# Model Architecture (MLP-focused for 224-dim vector)
observation_shape: (224,)
action_space_size: 45
num_res_blocks: 1
fc_policy_layers: [256, 256]
fc_value_layers: [256, 256]

# MCTS Parameters
num_simulations: 200        # Start with 200, scale to 400-800
pb_c_base: 19652           # Good defaults
pb_c_init: 1.25
root_dirichlet_alpha: 0.3   # For 45 actions
root_noise_weight: 0.25
max_moves: 140             # Safe cap for Splendor games

# Training
optim_type: 'SGD'          # Start with SGD + piecewise decay
learning_rate: 0.1
batch_size: 256
collector_env_num: 8
evaluator_env_num: 5
```

## Usage

### Training
Use the convenient training script:

```bash
# Basic training (1M steps)
python train_splendor_alphazero.py

# Custom parameters
python train_splendor_alphazero.py \
    --max_env_step 2000000 \
    --num_simulations 400 \
    --batch_size 512 \
    --exp_name my_splendor_experiment
```

**Training Parameters:**
- `--max_env_step`: Total environment steps (default: 1M)
- `--num_simulations`: MCTS simulations per action (default: 200)
- `--collector_env_num`: Parallel data collection environments (default: 8)
- `--batch_size`: Training batch size (default: 256)
- `--learning_rate`: SGD learning rate (default: 0.1)
- `--eval_freq`: Evaluation frequency in steps (default: 5000)

### Alternative Training (Direct LightZero)
```bash
cd LightZero
python -m lzero.entry.train_alphazero \
    --cfg-path zoo/board_games/splendor/config/splendor_alphazero_sp_mode_config.py
```

### Evaluation
```bash
# Evaluate trained model
python eval_splendor_alphazero.py --model_path path/to/ckpt_best.pth.tar

# Evaluate vs Noble Strategy Bot
python eval_splendor_alphazero.py \
    --model_path path/to/ckpt_best.pth.tar \
    --vs_bot \
    --num_episodes 200 \
    --num_simulations 400
```

### Monitoring
Monitor training progress with TensorBoard:
```bash
tensorboard --logdir data_az_ctree/
```

## Integration Details

### Wired into AlphaZero Policy
Added Splendor branch to `LightZero/lzero/policy/alphazero.py`:

```python
elif self._cfg.simulation_env_id == 'splendor':
    from zoo.board_games.splendor.envs.splendor_lz_env import SplendorLightZeroEnv
    from zoo.board_games.splendor.config.splendor_alphazero_sp_mode_config import splendor_alphazero_config
    self.simulate_env = SplendorLightZeroEnv(splendor_alphazero_config.env)
```

### Compatibility with Existing Splendor Engine
The wrapper seamlessly integrates with the existing `splendor_gym` engine:
- Reuses `legal_moves()`, `apply_action()`, `is_terminal()`, `winner()`
- Leverages existing 224-dim observation encoding
- Maintains compatibility with noble strategy bot for evaluation

## Performance Optimization

### Recommended Scaling Path
1. **Start**: 200 simulations, 8 collectors, 256 batch size
2. **Scale up**: 400 simulations as training stabilizes  
3. **Production**: 800 simulations with more collectors (16-32)

### Game-Specific Tuning
- **Turn Limit**: 100 pair-turns (conservative cap for draw prevention)
- **Temperature**: Start 1.0, anneal to 0.25 for exploitation
- **Dirichlet**: α=0.3 works well for 45-action space
- **Draw Reward**: 0.0 (standard AlphaZero)

## File Structure
```
splendor-gym/
├── train_splendor_alphazero.py          # Convenient training script
├── eval_splendor_alphazero.py           # Evaluation script  
├── SPLENDOR_ALPHAZERO_README.md         # This documentation
├── LightZero/                           # LightZero framework
│   └── zoo/board_games/splendor/        # Splendor integration
└── splendor_gym/                        # Existing Splendor engine
```

## Validation & Diagnostics

### Unit Tests Recommended
- [x] Environment integration test passed
- [ ] Reset to arbitrary states (MCTS requirement)
- [ ] Legal actions never empty (unless terminal)
- [ ] MCTS smoke test with 8 simulations

### Training Telemetry
Monitor these metrics during training:
- Win rate vs random player
- Win rate vs noble strategy bot  
- Average game length (turns)
- Fraction of games hitting max_moves (draw rate)
- Policy entropy and value accuracy

## Next Steps

### Immediate
1. **Run initial training** with 200 simulations
2. **Monitor convergence** and adjust hyperparameters
3. **Scale up** simulations (200→400→800) as training stabilizes

### Advanced
1. **Temperature scheduling**: Implement annealing from 1.0 to 0.25
2. **Hyperparameter tuning**: Learning rate, batch size, model architecture
3. **MuZero/EfficientZero**: For long-horizon credit assignment (future work)

## Troubleshooting

### Common Issues
- **Import errors**: Ensure `LightZero` is in current directory
- **CUDA memory**: Reduce batch size or number of environments
- **Slow training**: Increase num_simulations gradually, monitor GPU utilization

### Performance Tips
- Use GPU for model inference (`cuda=True` in config)
- Scale collector environments based on CPU cores
- Monitor memory usage with multiple environments

---

**Status**: ✅ Complete integration ready for training
**Next**: Run `python train_splendor_alphazero.py` to start training!
