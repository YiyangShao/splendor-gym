exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'step_timeout': None,
            'auto_reset': True,
            'reset_timeout': None,
            'retry_type': 'reset',
            'retry_waiting_time': 0.1,
            'shared_memory': False,
            'copy_on_get': True,
            'context': 'fork',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'reset_inplace': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict',
            'type': 'subprocess'
        },
        'stop_value': 1,
        'n_evaluator_episode': 5,
        'env_id': 'Splendor',
        'non_zero_sum': False,
        'battle_mode': 'self_play_mode',
        'battle_mode_in_simulation_env': 'self_play_mode',
        'bot_action_type': 'noble_strategy',
        'replay_path': None,
        'agent_vs_human': False,
        'prob_random_agent': 0,
        'prob_expert_agent': 0,
        'channel_last': False,
        'scale': True,
        'alphazero_mcts_ctree': False,
        'num_players': 2,
        'max_turns': 100,
        'cfg_type': 'SplendorLightZeroEnvDict',
        'type': 'splendor',
        'import_names': ['zoo.board_games.splendor.envs.splendor_lz_env'],
        'collector_env_num': 8,
        'evaluator_env_num': 5,
        'save_replay_gif': False,
        'replay_path_gif': './replay_gif'
    },
    'policy': {
        'model': {
            'observation_shape': (1, 224, 1),
            'num_res_blocks': 1,
            'num_channels': 256,
            'action_space_size': 45,
            'value_head_hidden_channels': [256, 256],
            'policy_head_hidden_channels': [256, 256]
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'resume_training': False
        },
        'collect': {
            'collector': {
                'cfg_type': 'AlphaZeroCollectorDict',
                'type': 'episode_alphazero',
                'import_names': ['lzero.worker.alphazero_collector']
            }
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'figure_path': None,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'type': 'alphazero',
                'import_names': ['lzero.worker.alphazero_evaluator'],
                'stop_value': 1,
                'n_episode': 5
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 1000000,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict',
                'save_episode': False
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'on_policy': False,
        'cuda': True,
        'multi_gpu': False,
        'bp_update_sync': True,
        'traj_len_inf': False,
        'torch_compile': False,
        'tensor_float_32': False,
        'sampled_algo': False,
        'gumbel_algo': False,
        'update_per_collect': 50,
        'replay_ratio': 0.25,
        'batch_size': 256,
        'optim_type': 'SGD',
        'learning_rate': 0.1,
        'weight_decay': 0.0001,
        'momentum': 0.9,
        'grad_clip_value': 10.0,
        'value_weight': 1.0,
        'collector_env_num': 8,
        'evaluator_env_num': 5,
        'piecewise_decay_lr_scheduler': True,
        'threshold_training_steps_for_final_lr': 500000,
        'manual_temperature_decay': False,
        'threshold_training_steps_for_final_temperature': 100000,
        'fixed_temperature_value': 1.0,
        'mcts': {
            'num_simulations': 200,
            'max_moves': 140,
            'root_dirichlet_alpha': 0.3,
            'root_noise_weight': 0.25,
            'pb_c_base': 19652,
            'pb_c_init': 1.25
        },
        'cfg_type': 'AlphaZeroPolicyDict',
        'type': 'alphazero',
        'import_names': ['lzero.policy.alphazero'],
        'mcts_ctree': False,
        'simulation_env_id': 'splendor',
        'simulation_env_config_type': 'self_play',
        'lr_piecewise_constant_decay': True,
        'threshold_training_steps_for_lr_decay': 50000,
        'lr_decay_ratio': 0.1,
        'entropy_weight': 0.0,
        'n_episode': 8,
        'eval_freq': 5000,
        'device': 'cpu'
    },
    'exp_name': 'data_az_ctree/debug_test_seed0_250825_211343',
    'seed': 0
}
