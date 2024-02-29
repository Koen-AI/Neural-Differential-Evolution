import ioh
import torch

from NeuralDifferentialEvolution import NeuralDifferentialEvolution
from Optimisation import Optimisee  # import optimisation_func #
from Args_Parser import parse_args
# from FCNet import FCNet
# from LSTMNet import LSTMNet, make_flat_bounds, dict_to_flat_array, flat_array_to_dict
from Networks import FCNet, LSTMNet, make_flat_bounds, dict_to_flat_array, flat_array_to_dict
from Demo import demo
import numpy as np
# from dr1_checkers import main


if __name__ == "__main__":
    args = parse_args()

    # add zeros to the shape to ensure the shape has 2 elements
    while len(args.shape) < 2:
        args.shape.append(0)

    if args.env == "checkers":
        # self.env = "ma_gym:Checkers-v0"
        if args.compression:
            filter_size = 24  # size of the compressed observation for one checkers agent
        else:
            filter_size = 47  # size of the regular observation for one checkers agent
    elif args.env == "switch":  # switch
        # env = "ma_gym:Switch2-v0"
        filter_size = 4  # size of the observation for one switch_v2 agent
    else:
        print("Error: Unexpected env!\nPlease select 'checkers' or 'switch'")
        exit()

    if args.lstm:
        netp = LSTMNet(in_size=filter_size, inter_1=args.shape[0], inter_2=args.shape[1], bias=args.bias)
        netq = LSTMNet(in_size=filter_size, inter_1=args.shape[0], inter_2=args.shape[1], bias=args.bias)
    else:
        netp = FCNet(filter_size=filter_size, inter_1=args.shape[0], inter_2=args.shape[1], bias=args.bias)
        netq = FCNet(filter_size=filter_size, inter_1=args.shape[0], inter_2=args.shape[1], bias=args.bias)

    d = netp.print_size(debug=True)

    # define the IOH-problem class
    optimisation_func = Optimisee(env=args.env, lstm=args.lstm, shape=args.shape, bias=args.bias,
                                  compression=args.compression) # compression=args.compression) FXIME
    '''
    if args.test_env:
        # Only used to test the opt func separately
        # Not a very elaborate test, not perfect, use at your own discretion
        p = dict_to_flat_array(netp.state_dict())
        q = dict_to_flat_array(netq.state_dict())
        W = []
        W.extend(p)
        W.extend(q)
        print("len W = ", len(W))
        optimisation_func(W)
        print("opt func passed!")
        netp.print_size(debug=True)
        p_dict = flat_array_to_dict(p, netp.shape, netp.labels)
        q_dict = flat_array_to_dict(q, netq.shape, netq.labels)
        assert p_dict['fc1.weight'][0][3] == netp.state_dict()['fc1.weight'][0][3]
        if args.shape[1] > 0:
            assert p_dict['fc2.weight'][2][3] == netp.state_dict()['fc2.weight'][2][3]
        if args.bias:
            assert p_dict['fc1.bias'][3] == netp.state_dict()['fc1.bias'][3]
        if args.shape[1] > 0:
            assert p_dict['fc2.weight'][4][7] == netp.state_dict()['fc2.weight'][4][7]
            if args.bias:
                assert p_dict['fc2.bias'][4] == netp.state_dict()['fc2.bias'][4]
        print("ditcts seem fine!")
        exit()
    '''

    if args.demo or args.video:
        print("A demo will soon begin!")
        demo(net0=netp, net1=netq, w_file=args.policy, env_str=args.env, compression=args.compression, video=args.video)
        exit()

    if args.method == "ModDE" or args.method == "modDE" or args.method == "modde" or args.method == "MDE":
        ioh.problem.wrap_real_problem(
            optimisation_func,  # Handle to the function
            name=args.env,  # Name to be used when instantiating
            optimization_type=ioh.OptimizationType.MIN,  # Specify that we want to minimize
            lb=-1 * args.bounds,  # The lower bound
            ub=args.bounds,  # The upper bound
        )

        NN_logger = ioh.logger.Analyzer(folder_name=args.dir, store_positions=True)
        NN_problem = ioh.get_problem(args.env, dimension=2*d)
        NN_problem.attach_logger(NN_logger)

        # Define the IOH Algorithm
        NN_algorithm = NeuralDifferentialEvolution(abs_bounds=args.bounds,
                                                   seed=args.seed,
                                                   popsize=args.popsize,
                                                   maxiter=args.maxiter,
                                                   F=args.F,
                                                   CR=args.CR)

        # Cheecky print to verify the algorithm is actually running
        print("pre:   ")
        NN_algorithm(NN_problem)      # run the algorithm on the problem
        print("post:  ")

    elif args.method == "RNG" or args.method == "RandomSearch" or args.method == "RS" or args.method == "random_search":
        max_score = -100
        #     # budget param  * population    * dimensionality
        iters = args.maxiter * args.popsize * d * 2
        lb = -1 * args.bounds
        ub = args.bounds

        for evals in range(iters):
            # create random lists for each agent in the range [-args.bounds, args.bounds]:
            lst_pq = args.bounds * 2 * np.random.rand(2*d) - args.bounds
            score = -1 * optimisation_func(lst_pq)  # *-1 because optimisation_func is designed for the minimizing ModDE

            if score > max_score:
                max_score = score
                pq_str = ""
                for weight in lst_pq:
                    pq_str += str(weight)
                    pq_str += " "
                pq_str = pq_str[:-1]

                print(evals, score, pq_str)

        # for evals
        pq_str = ""
        for weight in lst_pq:
            pq_str += str(weight)
            pq_str += " "
        print(evals, score, pq_str, ".fin")

    else:
        print("Please specify a valid search method:\nChose from ModDE, DR1, or RandomSearch/RNG")
