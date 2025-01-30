from __future__ import print_function, division
from MARL import MARL  
from common.utils import agg_double_list, copy_file, init_dir
from datetime import datetime

import argparse
import configparser
import sys
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env  # Import the highway environment

# Add the highway environment path
sys.path.append("../highway-env")


def parse_args():
    """
    Parses command-line arguments for training or evaluation.
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs.ini'
    
    parser = argparse.ArgumentParser(description='Train or evaluate a policy on an RL environment using MARL.')
    
    parser.add_argument('--base-dir', type=str, default=default_base_dir,
                        help="Base directory to save results.")
    parser.add_argument('--option', type=str, default='train',
                        help="'train' to train the model, 'evaluate' to evaluate the model.")
    parser.add_argument('--config-dir', type=str, default=default_config_dir,
                        help="Path to the configuration file.")
    parser.add_argument('--model-dir', type=str, default='',
                        help="Path to a pretrained model directory.")
    parser.add_argument('--evaluation-seeds', type=str, default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="Random seeds for evaluation, separated by commas.")
    
    return parser.parse_args()


def train(args):
    """
    Train the MARL model using the configuration provided in the config file.
    """
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # Create an experiment folder
    timestamp = datetime.now().strftime("%b-%d_%H_%M_%S")
    output_dir = os.path.join(base_dir, timestamp)
    dirs = init_dir(output_dir)
    copy_file(dirs['configs'])

    model_dir = args.model_dir if os.path.exists(args.model_dir) else dirs['models']

    # Load model and training configuration
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    training_strategy = config.get('MODEL_CONFIG', 'training_strategy')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    epsilon = config.getfloat('MODEL_CONFIG', 'epsilon')
    alpha = config.getfloat('MODEL_CONFIG', 'alpha')
    shared_network = config.getboolean('MODEL_CONFIG', 'shared_network')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')

    # Training configurations
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # Initialize environment
    env = gym.make('merge-multi-agent-v0')
    env_eval = gym.make('merge-multi-agent-v0')

    for param in ['seed', 'simulation_frequency', 'duration', 'policy_frequency', 'COLLISION_REWARD',
                  'HIGH_SPEED_REWARD', 'HEADWAY_COST', 'HEADWAY_TIME', 'MERGING_LANE_COST', 'traffic_density',
                  'safety_guarantee', 'n_step']:
        env.config[param] = config.get('ENV_CONFIG', param, fallback=None)

    env_eval.config.update(env.config)

    # Model initialization
    state_dim = env.n_s
    action_dim = env.n_a
    marl = MARL(env, state_dim, action_dim,
                memory_capacity=MEMORY_CAPACITY, roll_out_n_steps=ROLL_OUT_N_STEPS,
                reward_gamma=reward_gamma, reward_scale=reward_scale,
                actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                actor_lr=actor_lr, critic_lr=critic_lr,
                entropy_reg=ENTROPY_REG, batch_size=BATCH_SIZE,
                training_strategy=training_strategy, shared_network=shared_network,
                reward_type=reward_type, max_grad_norm=MAX_GRAD_NORM)

    # Load existing model if available
    marl.load(model_dir, train_mode=True)

    episodes, eval_rewards = [], []
    best_eval_reward = -float('inf')

    while marl.n_episodes < MAX_EPISODES:
        marl.explore()
        if marl.n_episodes >= EPISODES_BEFORE_TRAIN:
            marl.train()

        # Evaluate periodically
        if marl.episode_done and (marl.n_episodes % EVAL_INTERVAL == 0):
            rewards, *_ = marl.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)
            avg_reward = np.mean(rewards)
            print(f"Episode {marl.n_episodes}, Avg. Reward: {avg_reward:.2f}")
            episodes.append(marl.n_episodes)
            eval_rewards.append(avg_reward)

            if avg_reward > best_eval_reward:
                marl.save(dirs['models'], marl.n_episodes)
                best_eval_reward = avg_reward

        np.save(os.path.join(output_dir, 'eval_rewards.npy'), np.array(eval_rewards))

    # Save training progress and final model
    marl.save(dirs['models'], MAX_EPISODES)
    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Evaluation Rewards")
    plt.savefig(os.path.join(output_dir, "training_rewards.png"))
    plt.show()


def evaluate(args):
    """
    Evaluate a pretrained MARL model.
    """
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError("Pretrained model not found.")

    config = configparser.ConfigParser()
    config.read(os.path.join(args.model_dir, 'configs.ini'))

    env = gym.make('merge-multi-agent-v0')
    for param in ['seed', 'simulation_frequency', 'duration', 'policy_frequency', 'COLLISION_REWARD',
                  'HIGH_SPEED_REWARD', 'HEADWAY_COST', 'HEADWAY_TIME', 'MERGING_LANE_COST', 'traffic_density',
                  'safety_guarantee', 'n_step']:
        env.config[param] = config.get('ENV_CONFIG', param, fallback=None)

    marl = MARL(env, state_dim=env.n_s, action_dim=env.n_a)
    marl.load(args.model_dir)

    rewards, *_ = marl.evaluation(env, args.model_dir, eval_episodes=config.getint('TRAIN_CONFIG', 'EVAL_EPISODES'))
    print(f"Average Rewards: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    args = parse_args()
    if args.option == 'train':
        train(args)
    elif args.option == 'evaluate':
        evaluate(args)
    else:
        raise ValueError("Invalid option. Use 'train' or 'evaluate'.")

