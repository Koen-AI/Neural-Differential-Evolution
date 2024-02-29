import numpy as np
import torch
import time
import gym
import csv

from Networks import FCNet, LSTMNet, flat_array_to_dict
from Optimisation import compress_observation
from ma_gym.wrappers import Monitor


def demo(net0=None, net1=None, w_file=None, env_str="checkers", compression=False, video=False):
    t = 0  # timestep counter
    d = net0.print_size(debug=True)
    print(net0)

    if w_file != None:
        with open(w_file, 'r') as weights_file:
            flat_weights_1 = np.zeros(d, dtype=float)
            flat_weights_2 = np.zeros(d, dtype=float)
            for line in csv.reader(weights_file, delimiter=' '):
                for i in range(d):
                    flat_weights_1[i] = float(line[i + 2])
                    flat_weights_2[i] = float(line[i + 2 + d])

        state_dict_1 = flat_array_to_dict(flat_weights_1, net0.shape, net0.labels)
        state_dict_2 = flat_array_to_dict(flat_weights_2, net1.shape, net1.labels)

        net0.load_state_dict(state_dict_1)
        net1.load_state_dict(state_dict_2)

    with torch.no_grad():
        if env_str == "checkers":
            env = gym.make('ma_gym:Checkers-v0')
        else:
            env = gym.make('ma_gym:Switch2-v0')
        if video:
            env = Monitor(env, directory='recordings', force=True)
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        # time delays for rendering:
        start = 0.5  # 1
        end = 0.5  # 1
        interval = 0.5  # 1

        # Play one game:
        obs_n = env.reset()
        while not all(done_n):
            env.render()
            # Preprocess the observations:
            if env_str == "checkers":
                if compression:
                    compressed_obs0 = compress_observation(obs_n[0])
                    compressed_obs1 = compress_observation(obs_n[1])

                    obs_tensor0 = torch.tensor(compressed_obs0).float()
                    obs_tensor1 = torch.tensor(compressed_obs1).float()

                else:  # if not compression:
                    obs_tensor0 = torch.tensor(obs_n[0]).float()
                    obs_tensor1 = torch.tensor(obs_n[0]).float()
            else:
                obs = [obs_n[0][0], obs_n[0][1], obs_n[1][0], obs_n[1][1]]  # this is ugly but it works...

                obs_tensor0 = torch.tensor(obs).float()
                obs_tensor1 = torch.tensor(obs).float()

            # print("obs0 = ", obs_tensor0)
            # print("obs1 = ", obs_tensor1)
            # Get the actions from the networks:
            action_1 = np.argmax(net0.forward(obs_tensor0)).item()
            action_2 = np.argmax(net1.forward(obs_tensor1)).item()
            action = [action_1, action_2]

            # Take a step in the environment:
            obs_n, reward_n, done_n, info = env.step(action)
            print("t = ", t)
            t += 1
            print("act = ", action)
            print("reward_n = ", reward_n)
            ep_reward -= sum(reward_n)  # -= here because DE is a min. alg.

            # Make sure a human can see the actions being taken:
            time.sleep(interval+start)
            start = 0

        env.render()
        time.sleep(interval+end)
        env.close()


        print("total timesteps = ", t)
        print("tot_reward = ", ep_reward)