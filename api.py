from omegaconf import DictConfig
from jax.random import KeyArray

class BLOInstance:
    def __init__(self, cfg, rng) -> None:
        self.cfg = cfg
        self.rng = rng
    
    def get_upper_level_fn(self, ):
        raise NotImplementedError
    
    def get_lower_level_fn(self, ):
        raise NotImplementedError
    
class Method:
    def __init__(self, blo_instance: BLOInstance, cfg: DictConfig, rng: KeyArray) -> None:
        pass

    def value_and_grad_fn(self, forward_fn, params, rng):
        # the data generating process should be handled within this function
        raise NotImplementedError
    
    def plot_fn(self, forward_fn, params, rng):
        raise NotImplementedError
    
    def metric_fn(self, forward_fn, params, time_interval, rng):
        pass