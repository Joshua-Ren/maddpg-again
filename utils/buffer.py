import numpy as np
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True,
               other_pos_n = 1, other_neg_n = 1):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]

        # print('DEBUG',len(self.obs_buffs),self.obs_buffs[0][897].shape)
        # sampling obs in negative/positive directions (e.g. s_(t-x), s_(t+x) )
        pos_obs = [np.zeros((N,other_pos_n*5)) for i in range(self.num_agents)]
        neg_obs = [np.zeros((N,other_neg_n*5)) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            if i == 1:
                # 我不知道该怎么处理listener的观察，这里是忽略之
                continue
            for j, ind in enumerate(inds):

                if ind == 0:
                    pass
                elif ind <= other_neg_n - 1 :
                    # print('DEBUG',i,j,ind,self.obs_buffs[i][:ind])
                    neg_obs[i][j, -ind*5:] = np.concatenate(self.obs_buffs[i][:ind],axis=0)
                else:
                    # print(np.concatenate(self.obs_buffs[i][ind-other_neg_n:ind],axis=0).shape)
                    # print(neg_obs[i][j,:], self.obs_buffs[i][ind-other_neg_n:ind])
                    # print('DEBUG',i,j,ind,self.obs_buffs[i][ind-other_neg_n:ind])
                    neg_obs[i][j,:] = np.concatenate(self.obs_buffs[i][ind-other_neg_n:ind],axis=0)

                if ind == self.filled_i-1:
                    pass
                elif ind >= self.filled_i - other_pos_n:
                    pos_obs[i][j,:(self.filled_i-ind-1)*5] = np.concatenate(self.obs_buffs[i][ind+1:self.filled_i],axis=0)
                else:
                    pos_obs[i][j,:] = np.concatenate(self.obs_buffs[i][ind+1:ind+1+other_pos_n],axis=0)


        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(pos_ob) for pos_ob in pos_obs],
                [cast(neg_ob) for neg_ob in neg_obs]
                )

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
