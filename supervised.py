import numpy as np
import torch
import torch.nn as nn
import csv

from Networks import dict_to_flat_array, FCNet
from Demo import demo


def train_agent(cheat_file_str, max_iters=222222):
    # TODO use the next two lines to change between the switch and checkers cheatsheets:
    # agent = FCNet(inter_1=0, inter_2=0)  # checkers
    agent = FCNet(filter_size=4, inter_1=0, inter_2=0)  # switch

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

        obs_tensor = torch.tensor(obs_list)
        act_tensor = torch.tensor(act_list)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)

    for i in range(max_iters):
        pred_net = agent.forward(obs_tensor)
        optimizer.zero_grad()
        l3 = loss(pred_net, act_tensor)

        # if i % 222 == 0:
        #    print("loss = ", l3)
        l3.backward()
        optimizer.step()

    # print(dict_to_flat_array(agent.state_dict()))
    print("loss = ", l3)
    return agent


def evaluate_agent(agent, cheat_file_str, show_mistakes=False):
    obs_list = []
    act_list = []
    with open(cheat_file_str, 'r') as cheat_file:
        for line in csv.reader(cheat_file, delimiter=','):
            str_obs = line[:-1]
            obs = [float(i) for i in str_obs]
            act = int(line[-1])

            obs_list.append(obs)
            act_list.append(act)

        obs_tensor = torch.tensor(obs_list)

    pred_net = agent.forward(obs_tensor)
    turn = 0
    wrong = 0
    for pred in pred_net:
        action = torch.argmax(pred).item()
        # print(action)
        if action != act_list[turn]:
            if show_mistakes:
                print("Action ", action, " at turn ", turn + 1, " was wrong; should be: ", act_list[turn])
            # print("observation = ")
            wrong += 1
        turn += 1

    print("There were ", wrong, " mistakes, out of ", len(act_list), " moves!")

    return wrong


if __name__ == "__main__":
    torch.manual_seed(3)
    le_m = 10000

    # TODO use the next two lines to change between the switch and checkers cheatsheets:
    file_red = "switch_cheatsheet_red.txt"
    file_blue = "switch_cheatsheet_blue.txt"
    # file_red = "checkers_cheatsheet_red.txt"
    # file_blue = "checkers_cheatsheet_blue.txt"
    # train agents
    print("training red:")
    red_agent = train_agent(file_red, max_iters=le_m)
    red_agent.print_size(debug=True)
    print("training blue:")
    blue_agent = train_agent(file_blue, max_iters=le_m)
    agent_file_str = "agents.txt"

    # convert agent weights to str:
    agents_str = "0.0 0.0"
    red_list = dict_to_flat_array(red_agent.state_dict())
    blue_list = dict_to_flat_array(blue_agent.state_dict())
    for red in red_list:
        agents_str += str(red)
        agents_str += " "
    for blue in blue_list:
        agents_str += str(blue)
        agents_str += " "
    # write agent weights to file:
    with open(agent_file_str, 'w') as agent_file:
        agent_file.write(agents_str)

    evaluate_agent(red_agent, file_red)
    evaluate_agent(blue_agent, file_blue)

    # show some stats:
    red_array = dict_to_flat_array(red_agent.state_dict())
    print("red Upperbound = ", max(red_array))
    print("red Lowerbound = ", min(red_array))

    blue_array = dict_to_flat_array(blue_agent.state_dict())
    print("Upperbound = ", max(blue_array))
    print("Lowerbound = ", min(blue_array))

    # print(red_agent.state_dict())
    # show demo of trained agents:
    # input("press [enter] to start the demo")
    play_demo = input("Would you like to see a demo?\n")
    if play_demo:
        if play_demo[0] == 'Y' or play_demo[0] == 'y':
            demo(net0=red_agent, net1=blue_agent, env_str="switch")
            # demo(net0=red_agent, net1=blue_agent, env_str="checkers")
