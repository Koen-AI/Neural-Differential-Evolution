from scipy.optimize import differential_evolution
from modde import ModularDE
import numpy as np


class NeuralDifferentialEvolution:
    def __init__(self, seed=42, popsize=15, maxiter=1000, bound_corr='saturate', F=0.5, CR=0.5, abs_bounds=7.0):
        self.seed = seed
        self.popsize = popsize
        self.maxiter = maxiter
        self.bound_corr = bound_corr
        self.lshade = None
        self.F = F
        self.CR = CR
        self.abs_bounds = abs_bounds

    def __call__(self, func):
        nde_mod = ModularDE(func, base_sampler='uniform', mutation_base='rand', mutation_reference='rand',
                            F=np.array([self.F] * self.popsize * func.meta_data.n_variables),
                            CR=np.array([self.CR] * self.popsize * func.meta_data.n_variables),
                            lb=np.ones((func.meta_data.n_variables, 1)) * -1 * self.abs_bounds,
                            ub=np.ones((func.meta_data.n_variables, 1)) * self.abs_bounds,
                            bound_correction=self.bound_corr, crossover='bin', lpsr=True,
                            budget=self.popsize * func.meta_data.n_variables * self.maxiter,
                            lambda_=self.popsize * func.meta_data.n_variables,
                            memory_size=6, use_archive=True, init_stats=True,
                            adaptation_method_F=None, adaptation_method_CR=None)
        nde_mod.run()
