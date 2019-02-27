#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:40:03 2019

@author: s1583620
"""

import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios



USE_CUDA = False
env_id = 'simple_speaker_listener'
#env_id = 'simple_tag'
buffer_length = int(1e6)
n_episodes = int(25000)
n_exploration_eps = int(25000)
n_rollout_threads = 1
final_noise_scale = 0.0
init_noise_scale = 0.3      # The exploration noise will gradually decreased
episode_length = 25
batch_size = 64
steps_per_update = 100
save_interval = 1000


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

model_dir = Path('./models') / env_id / 'Test_model'
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if
                         str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run
log_dir = run_dir / 'logs'
os.makedirs(log_dir)
logger = SummaryWriter(str(log_dir))

torch.manual_seed(1024)
np.random.seed(1024)

env = make_parallel_env(env_id, n_rollout_threads, 1024, True)
maddpg = MADDPG.init_from_env(env, agent_alg='MADDPG', adversary_alg='MADDPG', 
                              tau=0.01, lr=0.01, hidden_dim=64)

replay_buffer = ReplayBuffer(buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

t = 0
#for ep_i in range(0, n_episodes, n_rollout_threads):
for ep_i in range(0, 10, 1):
    #print("Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + n_rollout_threads, n_episodes))
    obs = env.reset()
    maddpg.prep_rollouts(device='cpu')
    
    explr_pct_remaining = max(0, n_exploration_eps - ep_i) / n_exploration_eps
    maddpg.scale_noise(final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
    maddpg.reset_noise()
    
    for et_i in range(episode_length):
        # rearrange observations to be per agent, and convert to torch Variable
        torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                              requires_grad=False) for i in range(maddpg.nagents)]
        # get actions as torch Variables
        torch_agent_actions = maddpg.step(torch_obs, explore=True)
        # convert actions to numpy arrays
        agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
        # rearrange actions to be per environment
        actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
        next_obs, rewards, dones, infos = env.step(actions)
        
        
        noisy_agent_actions = []
        for i in range(len(agent_actions)):
            noise = np.random.rand(agent_actions[i].shape[0],agent_actions[i].shape[1])
            tmp = agent_actions[i]*0
            tmp_action = np.argmax(agent_actions[i]+noise*5)
            tmp[0][tmp_action] = 1.0
            noisy_agent_actions.append(tmp)
        
        replay_buffer.push(obs, agent_actions, rewards, next_obs, dones) # Here pushing observations
        obs = next_obs
        t += n_rollout_threads
        
        if (len(replay_buffer) >= batch_size and
            (t % steps_per_update) < n_rollout_threads):
            maddpg.prep_training(device='cpu')      # If use GPU, here change
            for u_i in range(n_rollout_threads):
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(batch_size, to_gpu=USE_CUDA)
                    maddpg.update(sample, a_i, logger=logger)
                maddpg.update_all_targets()
            maddpg.prep_rollouts(device='cpu')

        
    # Logging part    
    ep_rews = replay_buffer.get_average_rewards(episode_length * n_rollout_threads)
    for a_i, a_ep_rew in enumerate(ep_rews): 
        logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
    



































































