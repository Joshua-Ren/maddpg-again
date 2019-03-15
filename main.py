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
import csv
USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action, benchmark=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))




    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,
                                  noisy_sharing = True,
                                  noisy_SNR = config.noisy_SNR,
                                  game_id = config.env_id,
                                  est_ac = config.est_action)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    print('#########################################################################')
    print('Adversary using: ', config.adversary_alg, 'Good agent using: ',config.agent_alg,'\n')
    print('Noisy SNR is: ', config.noisy_SNR)
    print('#########################################################################')
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')
        if ep_i%5000==0:
            maddpg.lr *= 0.5
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            if et_i%4000 ==1:
                maddpg.lr *=0.4
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA, 
                                                      other_pos_n=config.L_sample, 
                                                      other_neg_n=config.M_sample)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                    
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            print("Episodes %i-%i of %i, rewards are: \n" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
            for a_i, a_ep_rew in enumerate(ep_rews):
                print('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

        '''
        # *** perform validation every 1000 episodes. i.e. run N=10 times without exploration ***
        if ep_i % config.validate_every_n_eps == config.validate_every_n_eps-1:
            # 假设只有一个env在跑
            episodes_stats = []
            info_for_one_env_among_timesteps = []
            print('*'*10,'Validation BEGINS','*'*10)
            for valid_et_i in range(config.run_n_eps_in_validation):
                obs = env.reset()
                maddpg.prep_rollouts(device='cpu')
                explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
                maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
                maddpg.reset_noise()

                curr_episode_stats = []
                for et_i in range(config.episode_length):
                    # rearrange observations to be per agent, and convert to torch Variable
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                          requires_grad=False)
                                 for i in range(maddpg.nagents)]
                    # get actions as torch Variables
                    torch_agent_actions = maddpg.step(torch_obs, explore=False)
                    # convert actions to numpy arrays
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                    # rearrange actions to be per environment
                    actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                    next_obs, rewards, dones, infos = env.step(actions)

                    info_for_one_env_among_timesteps.append(infos[0]['n'])

                    curr_episode_stats.append(infos[0]['n'])

                    obs = next_obs
                episodes_stats.append(curr_episode_stats)

            print('Summary statistics:')
            if config.env_id == 'simple_tag':
                # avg_collisions = sum(map(sum,info_for_one_env_among_timesteps))/config.run_n_eps_in_validation
                episodes_stats = np.array(episodes_stats)
                # print(episodes_stats.shape)
                # validation logging
                with open(f'{config.model_name}.log', 'a') as valid_logfile:
                    valid_logwriter = csv.writer(valid_logfile, delimiter=' ')
                    valid_logwriter.writerow(np.sum(episodes_stats,axis=(1,2)).tolist())
                avg_collisions = np.sum(episodes_stats)/episodes_stats.shape[0]
                print(f'Avg of collisions: {avg_collisions}')

            elif config.env_id == 'simple_speaker_listener':
                for i, stat in enumerate(info_for_one_env_among_timesteps):
                    print(f'ep {i}: {stat}')
            else:
                raise NotImplementedError
            print('*' * 10, 'Validation ENDS', '*' * 10)

        # *** END of VALIDATION ***
        '''
    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    valid_logfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1024, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=23000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--noisy_SNR", default = 50, type=float)
    parser.add_argument("--est_action", default = False)
    parser.add_argument("--L_sample", default = 1,type=int)
    parser.add_argument("--M_sample", default = 1,type=int)
    parser.add_argument("--validate_every_n_eps", default=100, type=int,
                        help="perform one validation after training n episodes")
    parser.add_argument("--run_n_eps_in_validation", default=10,
                        type=int, help="num of validation eps per N training eps")
    config = parser.parse_args()

    run(config)
