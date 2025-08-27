from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 5
num_simulations = 200  # Start with 200 as per roadmap
update_per_collect = 50
batch_size = 256
max_env_step = int(1e6)  # 1M steps for initial training
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

splendor_alphazero_config = dict(
    exp_name='data_az_ctree/splendor_sp-mode_alphazero_seed0',
    env=dict(
        env_id="Splendor",
        num_players=2,
        max_turns=100,  # Turn limit to prevent infinite games
        battle_mode='self_play_mode',
        bot_action_type='noble_strategy',  # Use noble strategy bot for evaluation
        channel_last=False,
        scale=True,  # Use normalized observations
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # ==============================================================
        # for the creation of simulation env
        agent_vs_human=False,
        prob_random_agent=0,
        prob_expert_agent=0,
        alphazero_mcts_ctree=mcts_ctree,
        save_replay_gif=False,
        replay_path_gif='./replay_gif',
        # ==============================================================
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        # ==============================================================
        # for the creation of simulation env
        simulation_env_id='splendor',
        simulation_env_config_type='self_play',
        # ==============================================================
        model=dict(
            observation_shape=(1, 224, 1),  # Reshape 224-dim vector to 3D for compatibility
            action_space_size=45,            # Total action space size
            # MLP-style model as per roadmap
            num_res_blocks=1,                # Start with 1 residual block
            num_channels=256,                # Hidden dimension
            # FC heads for policy and value
            value_head_hidden_channels=[256, 256],
            policy_head_hidden_channels=[256, 256],
        ),
        cuda=True,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',              # Start with SGD as per roadmap
        piecewise_decay_lr_scheduler=True,
        learning_rate=0.1,             # Start with 0.1 for SGD
        lr_piecewise_constant_decay=True,
        threshold_training_steps_for_lr_decay=int(5e4),
        lr_decay_ratio=0.1,
        grad_clip_value=10.0,
        value_weight=1.0,
        entropy_weight=0.0,
        n_episode=n_episode,
        eval_freq=int(5e3),            # Evaluate every 5k steps
        mcts=dict(
            num_simulations=num_simulations,
            pb_c_base=19652,           # Good defaults from roadmap
            pb_c_init=1.25,
            root_dirichlet_alpha=0.3,  # For 45 actions, 0.3 is reasonable
            root_noise_weight=0.25,
            max_moves=140,             # Splendor games often < 100 pair-turns, set safe cap
        ),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        # Temperature scheduling
        fixed_temperature_value=1.0,    # Start with 1.0, will anneal later
        manual_temperature_decay=False, # Keep fixed initially
    ),
)

splendor_alphazero_config = EasyDict(splendor_alphazero_config)
main_config = splendor_alphazero_config

splendor_alphazero_create_config = dict(
    env=dict(
        type='splendor',
        import_names=['zoo.board_games.splendor.envs.splendor_lz_env'],
    ),
    env_manager=dict(type='base'),  # Changed from 'subprocess' to avoid caching issues
    policy=dict(
        type='alphazero',
        import_names=['lzero.policy.alphazero'],
    ),
    collector=dict(
        type='episode_alphazero',
        import_names=['lzero.worker.alphazero_collector'],
    ),
    evaluator=dict(
        type='alphazero',
        import_names=['lzero.worker.alphazero_evaluator'],
    )
)
splendor_alphazero_create_config = EasyDict(splendor_alphazero_create_config)
create_config = splendor_alphazero_create_config

if __name__ == '__main__':
    from lzero.entry import train_alphazero
    train_alphazero([main_config, create_config], seed=0, max_env_step=max_env_step)
