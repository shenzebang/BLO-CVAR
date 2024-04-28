from api import SLOProblem, Distribution
import jax.random as random
import jax.numpy as jnp
import jax

def langevin_sample(num_iterations, epsilon, step_size, xy_init: jnp.ndarray, rng_0, score_fn):
    xy = xy_init
    rngs = random.split(rng_0, num_iterations)
    for rng in rngs:
        brownian_motion = random.normal(rng, (xy.shape[-1],)) 
        xy = xy - step_size * score_fn(xy) + jnp.sqrt(2.*epsilon)* brownian_motion * jnp.sqrt(step_size)
    return xy

langevin_sample = jax.vmap(langevin_sample, in_axes=[None, None, None, 0, 0, None])

class BLOLangevinSampler:
    def __init__(self, problem: SLOProblem, initial_distribution: Distribution, cfg) -> None:
        self.cfg = cfg
        self.batch_size = cfg.solver.langevin_sampler.batch_size
        self.step_size = cfg.solver.langevin_sampler.step_size
        self.number_of_steps = cfg.solver.langevin_sampler.number_of_steps
        self.temperature = cfg.solver.langevin_sampler.temperature
        self.initial_distribution = initial_distribution
        self.problem = problem
    
    def sample_fn(self, rng, param_UL):
        rng_init, rng_langevin = random.split(rng, 2)
        xy_init = self.initial_distribution.sample(rng_init, self.batch_size)
        rng_langevin = random.split(rng_langevin, self.batch_size)
        _value_fn = lambda x: self.problem.value_fn(param_UL, x)
        _score_fn = jax.grad(_value_fn)
        return langevin_sample(self.number_of_steps, self.temperature, self.step_size, 
                               xy_init, rng_langevin, _score_fn)
