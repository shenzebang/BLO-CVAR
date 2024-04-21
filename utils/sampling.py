from api import SLOProblem
import jax.random as random
import jax.numpy as jnp
import jax

def langevin_sample(num_iterations, epsilon, step_size, xy_init, rng_0, nabla_g):
    xy = xy_init
    rngs = random.split(rng_0, num_iterations)
    for rng in rngs:
        brownian_motion = random.normal(rng, (2,)) 
        xy = xy - step_size * nabla_g(xy) + jnp.sqrt(jnp.ones([])*2*epsilon)* brownian_motion * jnp.sqrt(jnp.ones([])*step_size)
    return xy

langevin_sample = jax.vmap(langevin_sample, in_axes=[None, None, None, 0, 0])

class LangevinSampler:
    def __init__(self, problem: SLOProblem, initial_distribution, cfg) -> None:
        self.problem = problem
        self.cfg = cfg
        self.batch_size = cfg.solver.cvarblosolver.langevin_sampler.batch_size
        self.step_size = cfg.solver.cvarblosolver.langevin_sampler.step_size
        self.number_of_steps = cfg.solver.cvarblosolver.langevin_sampler.number_of_steps
        self.temperature = cfg.solver.cvarblosolver.langevin_sampler.temperature
        self.initial_distribution = initial_distribution
    
    def sample_fn(self, rng):
        rng_init, rng_langevin = random.split(rng, 2)
        xy_init = self.initial_distribution.sample(rng_init, self.batch_size)
        rng_langevin = random.split(rng_langevin, self.batch_size)
        return langevin_sample(self.number_of_steps, self.temperature, self.step_size, xy_init, rng_langevin)
