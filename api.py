from omegaconf import DictConfig
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Dict, Tuple


TrainState = namedtuple("TrainState", "opt_state, UL_param, LL_param")
ProblemDimension = namedtuple("ProblemDimension", "UL_dim, LL_dim")

class Distribution(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(self, rng, batch_size):
        raise NotImplementedError

class SLOProblem(ABC):
    def __init__(self, min_or_max = 'max') -> None:
        self.min_or_max = min_or_max
        self.problem_dimension = None # an SLOProblem instance should specify the problem dimension.

    @abstractmethod
    def value_fn(self, theta, x):
        raise NotImplementedError

class BLOProblem(ABC):
    def __init__(self, 
                 cfg, rng) -> None:
        self.cfg = cfg
        self.rng = rng
        self.problem_UL = self.init_upper_level_problem_fn()
        self.problem_LL = self.init_lower_level_problem_fn()
        self.dim = self._get_problem_dimension_fn()
        
    def _get_problem_dimension_fn(self, ) -> ProblemDimension:
        if not isinstance(self.problem_UL.problem_dimension, ProblemDimension) or not isinstance(self.problem_LL.problem_dimension, ProblemDimension):
            raise ValueError("Please define the problem dimension of the upper and the lower level problems!")
        if self.problem_UL.problem_dimension != self.problem_LL.problem_dimension:
            raise ValueError("The upper and lower level problems must have the same problem dimension!")
        return self.problem_UL.problem_dimension

    @abstractmethod
    def init_upper_level_problem_fn(self, ) -> SLOProblem:
        raise NotImplementedError
    
    @abstractmethod
    def init_lower_level_problem_fn(self, ) -> SLOProblem:
        raise NotImplementedError
    
    
    
class BLOSolver(ABC):
    def __init__(self, blo_problem: BLOProblem, cfg: DictConfig, rng) -> None:
        self.blo_problem = blo_problem
        
    @abstractmethod
    def init_train_state(self, rng_init) -> TrainState:
        raise NotImplementedError

    @abstractmethod
    def step_fn(self) -> Tuple[TrainState, Dict]: 
        pass
    
    @abstractmethod
    def plot_fn(self, rng, ts: TrainState):
        raise NotImplementedError
    
    # @abstractmethod
    # def metric_fn(self, forward_fn, params, time_interval, rng):
    #     pass