# AlphaZero Training Fixes Summary

## Root Cause Analysis
The training was hanging because:
1. **Episodes never completed** - turn limit wasn't enforced
2. **Serialization failures** - SplendorState objects couldn't be tensorized  
3. **Missing required keys** - eval_episode_return expected by collector
4. **Environment caching** - subprocess manager cached old environment versions

## Key Fixes Applied

### 1. Turn Limit Enforcement ✅
**File**: `LightZero/zoo/board_games/splendor/envs/splendor_lz_env.py`
**Issue**: Episodes ran forever (turn_count increased but game_over stayed False)
**Fix**: Added turn limit check in step() method
```python
if self.state.turn_count >= self.max_turns:
    self.state = replace(self.state, game_over=True, turn_limit_reached=True)
```

### 2. SplendorState Serialization ✅  
**File**: `LightZero/zoo/board_games/splendor/envs/splendor_lz_env.py`
**Issue**: `TypeError: not support item type: <class 'splendor_gym.engine.state.SplendorState'>`
**Fix**: Modified current_state(), reset(), and step() to return numpy arrays
```python
# In current_state()
board_state = raw_obs.astype(np.float32)  # Instead of copy.deepcopy(self.state)

# In reset()
board_state, _ = self.current_state()  # Use serializable version

# In step() illegal action handler  
board_state, _ = self.current_state()  # Use serializable version
```

### 3. eval_episode_return Key ✅
**File**: `LightZero/zoo/board_games/splendor/envs/splendor_lz_env.py`  
**Issue**: `KeyError: 'eval_episode_return'` in AlphaZero collector
**Fix**: Added eval_episode_return to timestep info when episodes complete
```python
if done:
    scores = [player.prestige for player in self.state.players]
    if len(scores) >= 2:
        eval_episode_return = float(scores[0] - scores[1])
    else:
        eval_episode_return = float(scores[0])
    info['eval_episode_return'] = eval_episode_return
```

### 4. Environment Manager ✅
**File**: `LightZero/zoo/board_games/splendor/config/splendor_alphazero_sp_mode_config.py`
**Issue**: Subprocess environment manager cached old environment versions
**Fix**: Changed to base environment manager
```python
env_manager=dict(type='base')  # Instead of type='subprocess'
```

### 5. MCTS Performance Optimization ✅
**Issue**: 200 MCTS simulations too slow for development/testing
**Fix**: Reduced to configurable 10-50 simulations for reasonable speed

## Result
- ✅ Episodes complete properly (2-20 seconds each)
- ✅ Buffer receives data (1,880 transitions collected successfully)  
- ✅ Training iterations work (15-60 seconds each)
- ✅ Model checkpoints save correctly
- ✅ Loss values update showing learning progress

## Files Modified
1. `LightZero/zoo/board_games/splendor/envs/splendor_lz_env.py` - Main fixes
2. `LightZero/zoo/board_games/splendor/config/splendor_alphazero_sp_mode_config.py` - Config fix
3. `train_splendor_working.py` - Working training script with optimizations

## Training Time Estimates
- **Fast** (10 MCTS, 20 turns): ~4 hours for 1K iterations
- **Medium** (25 MCTS, 50 turns): ~7 hours for 1K iterations  
- **Production** (50 MCTS, 100 turns): ~12 hours for 1K iterations
