import time
import gym
import csv
import torch

from Optimisation import compress_observation


def get_action_list(cheat_file_str):
    action_list = []
    with open(cheat_file_str, 'r') as cheat_file:
        for line in csv.reader(cheat_file, delimiter=','):
            act = [int(i) for i in line]
            action_list.append(act)
            # if-else
        # for line
    return action_list


def generate(env_str="checkers", action_file="action_sheet.txt", write=False, readable=False):
    act_list = get_action_list(action_file)

    if env_str == "checkers":
        env = gym.make('ma_gym:Checkers-v0')
    else:
        env = gym.make('ma_gym:Switch2-v0')
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    # Play one game:
    obs_n = env.reset()

    # time delays for rendering:
    start = 1
    end = 1
    interval = 0.5

    if write:
        start = 0
        end = 0
        interval = 0

    # files to write to
    red_file_str = "switch_cheatsheet_red.txt"
    blue_file_str = "switch_cheatsheet_blue.txt"

    if readable:
        red_file_str = "switch_readable_red_cheatsheet.txt"
        blue_file_str = "switch_readable_blue_cheatsheet.txt"

    # dicts to detect double observations
    red_dict = {}
    blue_dict = {}

    index = 0
    while not all(done_n):
        env.render()
        print("shape = [", len(obs_n[0]), ", ", len(obs_n[1]), "]")
        if write:
            # Preprocess the observations:
            if env_str == "checkers":
                obs_r = compress_observation(obs_n[0])
                obs_b = compress_observation(obs_n[1])
                interval = 3

            else:  # elif env_str == "switch":
                obs = [obs_n[0][0], obs_n[0][1], obs_n[1][0], obs_n[1][1]]
                obs_r = obs
                obs_b = obs
                interval = 2

            # build the red observation:
            red_str = ""
            p = 0
            for entry in range(len(obs_r)):
                red_str += (str(obs_r[entry]))
                p += 1
                if readable and p % interval == 0:
                    red_str += "|"
                else:
                    red_str += ","

            # test if the same observation will always give the same action:
            if red_str in red_dict:
                if act_list[index][0] != red_dict[red_str]:  # Error message:
                    print("Error red act ", act_list[index][0], " != ", red_dict[red_str], " for obs:")
                    print(red_str)
                    print("in step: ", index)
                    env.render()
                    time.sleep(50)
                    exit()

            else:  # if red_str not in red_dict
                # add the right action for red
                red_dict[red_str] = act_list[index][0]
                red_str += str(act_list[index][0])
                red_str += "\n"

                # write to file:
                with open(red_file_str, 'a') as red_file:
                    red_file.write(red_str)

            # build the blue observation:
            blue_str = ""
            for entry in range(len(obs_b)):
                blue_str += (str(obs_b[entry]))
                p += 1
                if readable and p % interval == 0:
                    blue_str += "|"
                else:
                    blue_str += ","

            # test if the same observation will always give the same action:
            if blue_str in blue_dict:
                if act_list[index][1] != blue_dict[blue_str]:  # Error message:
                    print("Error blue act ", act_list[index][1], " != ", blue_dict[blue_str], " for obs:")
                    print(blue_str)
                    print("in step: ", index)
                    env.render()
                    time.sleep(50)
                    exit()
            else:  # if blue_str not in blue_dict
                # add the right action for  blue
                blue_dict[blue_str] = act_list[index][1]
                blue_str += str(act_list[index][1])
                blue_str += "\n"

                # write to file:
                with open(blue_file_str, 'a') as blue_file:
                    blue_file.write(blue_str)
        # if write to file

        # Take a step in the environment:
        if write:
            print("obs red = ", red_str)
            print("obs blue = ", blue_str)
        print("act = ", act_list[index])

        obs_n, reward_n, done_n, info = env.step(act_list[index])
        # print("\n\nobs_n = ", obs_n)

        # print("step = ", index)
        # print("reward_n = ", reward_n)
        ep_reward -= sum(reward_n)  # -= here because DE is a min. alg.

        # Make sure a human can see the actions being taken:
        time.sleep(interval + start)  # This is not desired when training!
        start = 0
        index += 1
    env.render()
    time.sleep(interval + end)
    env.close()

    print("tot_reward = ", ep_reward)


def retreive_lists(cheat_file_str):
    obs_list = []
    act_list = []
    with open(cheat_file_str, 'r') as cheat_file:
        for line in csv.reader(cheat_file, delimiter=','):
            str_obs = line[:-1]
            obs = [float(i) for i in str_obs]
            # blue_observation = torch.tensor(lst_obs)
            act = int(line[-1])
            # print("obs = ", obs)
            # print("act = ", act)
            # blue_tuple = (blue_observation, blue_action)
            obs_list.append(obs)
            act_list.append(act)

    # obs_tensor = torch.tensor(obs_list)

    return obs_list, act_list


def list_equal(obs1, obs2):
    if len(obs1) != len(obs2):
        return False

    for i in range(len(obs1)):
        if obs1[i] != obs2[i]:
            return False
    return True


def test_cheatsheet(env_str="switch", cheat_file_str_red="switch_cheatsheet_red.txt",
                    cheat_file_str_blue="switch_cheatsheet_blue.txt", no_delay=False):
    obs_list_red, act_list_red = retreive_lists(cheat_file_str_red)
    obs_list_blue, act_list_blue = retreive_lists(cheat_file_str_blue)

    if env_str == "checkers":
        env = gym.make('ma_gym:Checkers-v0')
    else:
        env = gym.make('ma_gym:Switch2-v0')
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    # Play one game:
    obs_n = env.reset()

    # time delays for rendering:
    start = 1
    end = 1
    interval = 0.5

    if no_delay:
        start = 0
        end = 0
        interval = 0

    index = 0
    while not all(done_n):
        env.render()
        print("raw obs = ", obs_n)
        print("shape = [", len(obs_n[0]), ", ", len(obs_n[1]), "]")
        if env_str == "checkers":
            obs_r = compress_observation(obs_n[0])
            obs_b = compress_observation(obs_n[1])
        elif env_str == "switch":
            obs = [obs_n[0][0], obs_n[0][1], obs_n[1][0], obs_n[1][1]]
            obs_r = obs
            obs_b = obs
        print("processed = ", obs_r)
        print("processed = ", obs_b)


        life_r = True
        life_b = True

        red_index = -1
        for obs in obs_list_red:
            red_index += 1
            if list_equal(obs, obs_r):
                life_r = False
                break

        blue_index = -1
        for obs in obs_list_blue:
            blue_index += 1
            if list_equal(obs, obs_b):
                life_b = False
                break
        # fin

        if life_r or life_b:
            print("error observations don't match")
            exit()

        # Take a step in the environment:
        print("act = ", [act_list_red[red_index], act_list_blue[blue_index]])

        obs_n, reward_n, done_n, info = env.step([act_list_red[red_index], act_list_blue[blue_index]])
        # print("\n\nobs_n = ", obs_n)

        # print("step = ", index)
        # print("reward_n = ", reward_n)
        ep_reward -= sum(reward_n)  # -= here because DE is a min. alg.

        # Make sure a human can see the actions being taken:
        time.sleep(interval + start)  # This is not desired when training!
        start = 0
        index += 1
    env.render()
    time.sleep(interval + end)
    env.close()

    print("tot_reward = ", ep_reward)


if __name__ == "__main__":
    # generate(write=True, readable=False, env_str="switch", action_file="switch_action_sheet.txt")
    # generate(write=False)   # Change this to write is True to generate new data
    test_cheatsheet()
    time.sleep(0)           # Right now this is a glorified demo with a fixed strategy...
