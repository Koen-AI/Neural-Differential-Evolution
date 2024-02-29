import numpy as np
import torch
import gym

from Networks import FCNet, LSTMNet, flat_array_to_dict


def compress_observation(obs):
    comp_obs = np.zeros(24, dtype=float)

    # Compress the 1-hot encoding to numerical representation:
    # 22 is intentionally missing because the agent itself is always there!
    for i, ci in zip([2, 7, 12, 17, 27, 32, 37, 42], [0, 3, 6, 9, 12, 15, 18, 21]): # [2, 5, 8, 11, 14, 17, 20, 23]):
        # if obs[i] == 1 (but floats were used so...)
        if obs[i] > 0.5:  # This is a lemon!
            comp_obs[ci] = 1
        elif obs[i+1] > 0.5:  # This is an apple!
            comp_obs[ci+1] = 1
        elif obs[i+2] > 0.5 or obs[i+3] > 0.5:  # This is an agent!
            comp_obs[ci+2] = 1
        elif obs[i+4] > 0.5:  # This is a wall! (Should never happen!)
            print("Let there be walls!")
    return comp_obs


class Optimisee():
    def __init__(self, env="ma_gym:Checkers-v0", lstm=False, shape=[0, 0], bias=False, compression=False):
        self.bias = bias
        self.compression = compression
        # initialize the networks and environments:
        if env == "checkers":
            self.env = "ma_gym:Checkers-v0"
            if self.compression:
                self.filter_size = 24  # size of the compressed observation for one checkers agent
            else:
                self.filter_size = 47  # size of the regular observation for one checkers agent
        else:  # switch
            self.env = "ma_gym:Switch2-v0"
            self.filter_size = 4  # size of the observation for one switch_v2 agent

        if lstm:
            self.net0 = LSTMNet(in_size=self.filter_size, inter_1=shape[0], inter_2=shape[1], bias=self.bias)
            self.net1 = LSTMNet(in_size=self.filter_size, inter_1=shape[0], inter_2=shape[1], bias=self.bias)
        else:
            self.net0 = FCNet(filter_size=self.filter_size, inter_1=shape[0], inter_2=shape[1], bias=self.bias)
            self.net1 = FCNet(filter_size=self.filter_size, inter_1=shape[0], inter_2=shape[1], bias=self.bias)

        self.shape = self.net0.shape
        self.cuttoff = self.net0.print_size()

    def __call__(self, weights: np.array):
        with torch.no_grad():
            env = gym.make(self.env)
            done_n = [False for _ in range(env.n_agents)]
            ep_reward = 0

            # Initialise the weights of the networks:
            weights0 = weights[:self.cuttoff]
            weights1 = weights[self.cuttoff:2 * self.cuttoff]

            state_dict0 = flat_array_to_dict(weights0, self.shape, self.net0.labels)
            state_dict1 = flat_array_to_dict(weights1, self.shape, self.net1.labels)

            self.net0.load_state_dict(state_dict0)
            self.net1.load_state_dict(state_dict1)

            # Play one game:
            obs_n = env.reset()
            while not all(done_n):
                # Preprocess the observations:
                if self.env == "ma_gym:Checkers-v0":
                    if self.compression:
                        compressed_obs0 = compress_observation(obs_n[0])
                        compressed_obs1 = compress_observation(obs_n[1])

                        obs_tensor0 = torch.tensor(compressed_obs0).float()
                        obs_tensor1 = torch.tensor(compressed_obs1).float()

                    else:  # if not self.compression:
                        obs_tensor0 = torch.tensor(obs_n[0]).float()
                        obs_tensor1 = torch.tensor(obs_n[0]).float()

                else:  # elif self.env == "ma_gym:Switch2-v0"
                    obs = [obs_n[0][0], obs_n[0][1], obs_n[1][0], obs_n[1][1]]  # this is ugly but it works...

                    obs_tensor0 = torch.tensor(obs).float()
                    obs_tensor1 = torch.tensor(obs).float()

                # Get the actions from the networks:
                action0 = torch.argmax(self.net0.forward(obs_tensor0)).item()
                action1 = torch.argmax(self.net1.forward(obs_tensor1)).item()
                action = [action0, action1]

                # Take a step in the environment:
                obs_n, reward_n, done_n, info = env.step(action)
                ep_reward -= sum(reward_n)  # -= here because DE is a min. alg.
            # while not done
            env.close()
        # with torch.nograd()
        return ep_reward
