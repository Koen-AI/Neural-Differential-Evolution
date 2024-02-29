import numpy as np
import torch.nn as nn
import torch.cuda
import torch
from collections import OrderedDict


class FCNet(nn.Module):
    def __init__(self, filter_size=24, inter_1=0, inter_2=0, out_actions=5, bias=False):
        super().__init__()
        self.filter_size = filter_size
        self.inter_1 = inter_1
        self.inter_2 = inter_2
        self.out_actions = out_actions
        self.bias = bias

        if inter_2 > 0 and bias:  # 2 hidden layers, with bias:
            self.shape = [[self.inter_1, self.filter_size], [self.inter_1], [self.inter_2, self.inter_1],
                          [self.inter_2],
                          [self.out_actions, self.inter_2], [self.out_actions]]
            self.num_params = self.shape[0][0] * self.shape[0][1] + self.shape[1][0] \
                            + self.shape[2][0] * self.shape[2][1] + self.shape[3][0] \
                            + self.shape[4][0] * self.shape[4][1] + self.shape[5][0]
            self.labels = ["fc1.weight", "fc1.bias",
                           "fc2.weight", "fc2.bias",
                           "fc3.weight", "fc3.bias"]
            self.fc1 = nn.Linear(filter_size, self.inter_1, bias=self.bias)
            self.fc2 = nn.Linear(self.inter_1, self.inter_2, bias=self.bias)
            self.fc3 = nn.Linear(self.inter_2, self.out_actions, bias=self.bias)

        elif inter_2 > 0 and not bias:  # 2 hidden layers, no bias:
            self.shape = [[self.inter_1, self.filter_size],
                          [self.inter_2, self.inter_1],
                          [self.out_actions, self.inter_2]]
            self.num_params = self.shape[0][0] * self.shape[0][1] \
                            + self.shape[1][0] * self.shape[1][1] \
                            + self.shape[2][0] * self.shape[2][1]
            self.labels = ["fc1.weight",
                           "fc2.weight",
                           "fc3.weight",]
            self.fc1 = nn.Linear(filter_size, self.inter_1, bias=self.bias)
            self.fc2 = nn.Linear(self.inter_1, self.inter_2, bias=self.bias)
            self.fc3 = nn.Linear(self.inter_2, self.out_actions, bias=self.bias)

        elif inter_1 > 0 and bias:  # 1 hidden layer, with bias:
            self.shape = [[self.inter_1, self.filter_size], [self.inter_1],
                          [self.out_actions, self.inter_1], [self.out_actions]]

            self.labels = ["fc1.weight", "fc1.bias",
                           "fc2.weight", "fc2.bias"]
            self.num_params = self.shape[0][0] * self.shape[0][1] + self.shape[1][0] \
                            + self.shape[2][0] * self.shape[2][1] + self.shape[3][0]
            self.fc1 = nn.Linear(filter_size, self.inter_1, bias=self.bias)
            self.fc2 = nn.Linear(self.inter_1, self.out_actions, bias=self.bias)

        elif inter_1 > 0 and not bias:  # 1 hidden layer, with bias:
            self.shape = [[self.inter_1, self.filter_size],
                          [self.out_actions, self.inter_1]]

            self.labels = ["fc1.weight",
                           "fc2.weight"]
            self.num_params = self.shape[0][0] * self.shape[0][1] \
                            + self.shape[1][0] * self.shape[1][1]
            self.fc1 = nn.Linear(filter_size, self.inter_1, bias=self.bias)
            self.fc2 = nn.Linear(self.inter_1, self.out_actions, bias=self.bias)

        elif self.bias:  # no hidden layer, with bias:
            self.shape = [[self.out_actions, self.filter_size], [self.out_actions]]

            self.labels = ["fc1.weight", "fc1.bias"]
            self.num_params = self.shape[0][0] * self.shape[0][1] + self.shape[1][0]
            self.fc1 = nn.Linear(filter_size, self.out_actions, bias=self.bias)

        else:  # no hidden layer, no bias:
            self.shape = [[self.out_actions, self.filter_size]]

            self.labels = ["fc1.weight"]
            self.num_params = self.shape[0][0] * self.shape[0][1]
            self.fc1 = nn.Linear(filter_size, self.out_actions, bias=self.bias)

    def print_size(self, debug=False):
        if debug:
            print("This network has ", self.num_params, " total parameters")
            print("The network shape is defined by: ", self.shape)
            print(self.labels)
        return self.num_params

    def __call__(self, inn):
        return self.forward(inn)

    def forward(self, inn):
        outt = self.fc1(inn)
        if self.inter_1 <= 0: # no hidden layers:
            return outt

        # 1+ hidden layers:
        outt = torch.sigmoid(outt)
        outt = self.fc2(outt)
        if self.inter_2 <= 0:  # only 1 hidden layer:
            return outt

        # 2 hidden layers:
        outt = torch.sigmoid(outt)
        outt = self.fc3(outt)
        return outt


class LSTMNet(nn.Module):
    # Bias is not yet implemented into the LSTMNet framework
    def __init__(self, in_size=24, inter_1=8, inter_2=8, out_actions=5, bias=False):
        super().__init__()
        # This architecture must have at least two hidden layers...
        assert inter_1 > 0
        assert inter_2 > 0
        self.in_size = in_size
        self.inter_1 = inter_1
        self.inter_2 = inter_2
        self.out_actions = out_actions
        self.bias = bias
        if self.bias:
            self.shape = [[self.inter_1, self.in_size],
                          [self.inter_1],
                          [self.inter_2 * 4, self.inter_1],
                          [self.inter_2 * 4, self.inter_2],
                          [self.inter_2 * 4],
                          [self.inter_2 * 4],
                          [self.out_actions, self.inter_2],
                          [self.out_actions]
                          ]
            self.labels = ["fc1.weight", "fc1.bias",
                           "lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
                           "fc2.weight", "fc2.bias"]

            self.num_params = self.shape[0][0] * self.shape[0][1] +\
                              self.shape[1][0] + \
                              self.shape[2][0] * self.shape[2][1] + \
                              self.shape[3][0] * self.shape[3][1] + \
                              self.shape[4][0] + \
                              self.shape[5][0] + \
                              self.shape[6][0] * self.shape[6][1] + \
                              self.shape[7][0]
        else: # if no bias:
            self.shape = [[self.inter_1, self.in_size],
                          [self.inter_2 * 4, self.inter_1],
                          [self.inter_2 * 4, self.inter_2],
                          [self.out_actions, self.inter_2]
                          ]
            self.labels = ["fc1.weight",
                           "lstm.weight_ih_l0", "lstm.weight_hh_l0",
                           "fc2.weight"]

            self.num_params = self.shape[0][0] * self.shape[0][1] + \
                              self.shape[1][0] * self.shape[1][1] + \
                              self.shape[2][0] * self.shape[2][1] + \
                              self.shape[3][0] * self.shape[3][1]

        self.fc1 = nn.Linear(in_size, self.inter_1, bias=self.bias)
        self.lstm = nn.LSTM(self.inter_1, self.inter_2, num_layers=1, batch_first=False, bias=self.bias)
        self.fc2 = nn.Linear(self.inter_2, self.out_actions, bias=self.bias)  # for a net with 1 hidden layer!

    def print_size(self, debug=False):
        if debug:
            print("This network has ", self.num_params, " total parameters")
            print("The network shape is defined by: ", self.shape)
        return self.num_params

    def __call__(self, a, inn):
        self.flat_array_to_self_dict(a, self.shape, self.labels)
        return self.forward(inn)

    def forward(self, inn):
        h0 = torch.zeros(1, 1, self.inter_2)
        c0 = torch.zeros(1, 1, self.inter_2)

        outt = self.fc1(inn)
        outt = torch.sigmoid(outt)
        outt = np.reshape(outt, (1, 1, self.inter_1))
        outt, (h0, c0) = self.lstm(outt, (h0, c0))

        outt = self.fc2(outt)

        return outt


def dict_to_flat_array(d):
    # print("conversion D2A")
    a = []
    for _, value in d.items():
        a.extend(np.ndarray.flatten(np.array(value.to("cpu"))))
    # print("converted")
    return np.ndarray.flatten(np.array(a))


def flat_array_to_dict(a, shape, labels):
    # print("conversion A2D")
    d = OrderedDict()
    a_index = 0                                                                        # counter for index in flat array
    # Loop over all pairs of shapes:
    '''
    print("Flat array to dict")
    print("i in range: ", range(len(shape)))
    print("all labels: ", labels)
    print("shape: ", shape)
    # '''
    for i in range(0, len(shape)):

        layer_name = labels[i]

        # Create the weights array:
        layer_array = np.zeros(shape[i], dtype=float)
        # Loop over weight shapes: should be 2-dimensional.

        if "weight" in layer_name:
            for k in range(len(layer_array)):
                for j in range(len(layer_array[k])):
                    layer_array[k][j] = a[a_index]
                    a_index += 1
                # for j
            # for k
        elif "bias" in layer_name:
            for m in range(len(layer_array)):
                layer_array[m] = a[a_index]
                a_index += 1
            # for m
        else:
            print("error in flat_array_to_dict")
            exit(1)
        layer_tensor = torch.tensor(layer_array, requires_grad=False)
        d[layer_name] = layer_tensor
    # for i
    return d


def make_flat_bounds(shape, labels):
    # print("making bounds")
    # Define the weight bounds:
    w_min = 0.0
    w_max = 1.0
    w = (w_min, w_max)

    # Define the bias bounds:
    b_min = -5.0
    b_max = 5.0
    b = (b_min, b_max)

    bounds = []
    '''
    print("Make flat bounds")
    print("i in range: ", range(len(shape)))
    print("all labels: ", labels)
    print("shape: ", shape)
    # '''
    for i in range(len(shape)):
        # print(labels[i])
        if "weight" in labels[i]:
            # Loop over weight shapes: should be 2-dimensional.
            for k in range(shape[i][0]):
                for j in range(shape[i][1]):
                    bounds.append(w)
                # for j
            # for k
        elif "bias" in labels[i]:
            # Loop over bounds shapes: should be 1-dimensional.
            for m in range(shape[i][0]):
                bounds.append(b)
            # for m
        else:
            print("error in make_flat_bounds")
            exit(1)
    # for i
    return bounds
