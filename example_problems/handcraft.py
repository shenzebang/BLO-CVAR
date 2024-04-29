from api import BLOProblem, SLOProblem, ProblemDimension
import jax.numpy as jnp
import jax 

def g(xy: jnp.ndarray):
    assert xy.ndim == 1
    x, y = xy[0], xy[1]
    conditions = [
        jnp.logical_and(y > jnp.sin(x), jnp.logical_and(0 < x,  x < 3*jnp.pi)),
        jnp.logical_and(x < 0, y > 0),
        jnp.logical_and(jnp.logical_and(-3*jnp.pi < y, y < 0), x < jnp.sin(y)),
        jnp.logical_and(y < -3*jnp.pi, x < 0),
        jnp.logical_and(y < -jnp.sin(x) - 3*jnp.pi, jnp.logical_and(0 < x, x <3*jnp.pi)),
        jnp.logical_and(y < -3*jnp.pi, x > 3*jnp.pi),
        jnp.logical_and(jnp.logical_and(-3*jnp.pi < y, y < 0), x > -jnp.sin(y) + 3*jnp.pi),
        jnp.logical_and(y > 0, x > 3*jnp.pi),
    ]
    functions = [
        (y - jnp.sin(x))**2,
        (y - x)**2,
        (x - jnp.sin(y))**2,
        (y + x + 3*jnp.pi)**2,
        (y + jnp.sin(x) + 3*jnp.pi)**2,
        (y - x + 6*jnp.pi)**2,
        (x + jnp.sin(y) - 3*jnp.pi)**2,
        (x - 3*jnp.pi + y)**2,
        jnp.zeros([])
    ]
    return jnp.piecewise(jnp.zeros([]), conditions, functions)

class HandCraftUpper(SLOProblem):
    def __init__(self, min_or_max='max') -> None:
        super().__init__(min_or_max)
        self.problem_dimension = ProblemDimension(UL_dim=2, LL_dim=2)

    def value_fn(self, theta, x):
        if x.ndim != 1:
            raise ValueError(f"x should be a 1-D array, but got {x.ndim} array as input.")
        return jnp.sum(x ** 2, axis=-1) 
    # + jnp.sum(theta ** 2, axis=-1)
    
class HandCraftLower(SLOProblem):
    def __init__(self, min_or_max='min') -> None:
        super().__init__(min_or_max)
        self.problem_dimension = ProblemDimension(UL_dim=2, LL_dim=2)

    def value_fn(self, theta, x):
        if x.ndim != 1:
            raise ValueError(f"x should be a 1-D array, but got {x.ndim} array as input.")
        return g(x - theta)


class HandCraft(BLOProblem):
    # def __init__(self, cfg, rng) -> None:
    #     super().__init__(cfg, rng)
        
    def init_upper_level_problem_fn(self, ):
        return HandCraftUpper()
    
    def init_lower_level_problem_fn(self, ):
        return HandCraftLower()