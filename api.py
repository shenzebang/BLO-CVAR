from omegaconf import DictConfig
from jax.random import KeyArray
from collections import namedtuple

TrainState = namedtuple("TrainState", "opt_state, UL_param, LL_param")
class SLOProblem:
    def __init__(self, min_or_max = 'max') -> None:
        self.min_or_max = min_or_max
        pass

    def value_fn(self):
        pass

class BLOProblem:
    def __init__(self, 
                 cfg, rng) -> None:
        self.upper_level_problem = self.init_upper_level_problem()
        self.lower_level_problem = self.init_lower_level_problem()
        self.cfg = cfg
        self.rng = rng

    def init_upper_level_problem(self, ) -> SLOProblem:
        raise NotImplementedError
    
    def init_lower_level_problem(self, ) -> SLOProblem:
        raise NotImplementedError
    
    
    
class BLOSolver:
    def __init__(self, blo_problem: BLOProblem, cfg: DictConfig, rng: KeyArray) -> None:
        self.blo_problem = blo_problem
        

    def init_train_state(self, rng_init) -> TrainState:
        raise NotImplementedError

    def step_fn(self) -> TrainState: 
        pass
    
    def plot_fn(self, forward_fn, params, rng):
        raise NotImplementedError
    
    def metric_fn(self, forward_fn, params, time_interval, rng):
        pass