from api import BLOInstance

def upper_level_fn(params):
    # f(theta, x)
    return 0

def lower_level_fn(params):
    # g(theta, x)
    return 0

class HandCraft1(BLOInstance):
    def __init__(self, cfg, rng) -> None:
        super().__init__(cfg, rng)
        
    def get_upper_level_fn(self, ):
        raise NotImplementedError
    
    def get_lower_level_fn(self, ):
        raise NotImplementedError