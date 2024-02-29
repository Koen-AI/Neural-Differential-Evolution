import argparse

# parses arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural Differential Evolution")
    # ======================= I/O ======================= #
    parser.add_argument('--dir', type=str, default="./results/run",
                        help="The IOH-profiler logs will be stored at this location")

    # ============= Neural Network settings ============= #
    parser.add_argument('--policy', type=str, default="weights.txt",
                        help="Path to the weights of a neural network")
    parser.add_argument('--lstm', action='store_true',
                        help='When this flag is true a network with an LSTM will be trained')
    parser.add_argument('--env', type=str, default="checkers", choices=["checkers", "switch"],
                        help="Environment to train on.")
    parser.add_argument('--shape', nargs='+', type=int, default=[0, 0],
                        help="The size of the hidden layers of the network. Up to two hidden layers are supported.")
    parser.add_argument('--bias', action='store_true',
                        help='Add biases to the layers of the neural network. Turned off by default')

    # ========= Differential Evolution settings ========= #
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    #parser.add_argument('--workers', type=int, default=1,
    #                    help='Number of parallel workers. Default is 1.')
    parser.add_argument('--popsize', default=15, type=int,
                        help='Size of the population for DE')
    parser.add_argument('--maxiter', default=1000, type=int,
                        help='Number of iterations to run DE, will be multiplied by popsize+1 and dimensionality internally by Scipy DE')
    parser.add_argument('--bounds', default=7.0, type=float,
                        help='The absolute value of the upper and lower bounds for the search space')
    parser.add_argument('--F', default=0.5, type=float,
                        help='The absolute value of the upper and lower bounds for the search space')
    parser.add_argument('--CR', default=0.5, type=float,
                        help='The absolute value of the upper and lower bounds for the search space')



    # ======= Settings when not training an agent ======= #
    parser.add_argument('--method', default="ModDE", choices=["ModDE", "modDE", "modde", "MDE", "RandomSearch", "RNG", "RS", "random_search"],
                        help="Optimization method, chose from modDE or RandomSearch")
    parser.add_argument('--compression', action='store_true',
                        help='Compress the observations for Checkers when training the agent')
    parser.add_argument('--random_search', action='store_true',
                        help='Perform random search instead of differential evolution')
    parser.add_argument('--demo', action='store_true',
                        help='Perform a demonstration instead of training cycle')
    parser.add_argument('--video', action='store_true',
                        help='Create a video instead of performing training cycle')
    # parser.add_argument('--test_env', action='store_true',
    #                     help="performs a single test iteration of the objective function")

    return parser.parse_args()
