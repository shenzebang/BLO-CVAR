from api import SLOProblem, Distribution
import jax.random as random
import jax.numpy as jnp
import jax
from jax import lax
from jax import jit
from functools import partial


def ula_kernel(rng, param, score_fn, dt, epsilon):
    rng, subkey = random.split(rng)
    paramGrad = score_fn(param)
    param = param - dt*paramGrad + jnp.sqrt(2*dt*epsilon)*random.normal(key=subkey, shape=(param.shape))
    return rng, param

# @partial(jit, static_argnums=(1,2,3,4))
def ula_sampler_full_jax_jit(rng, score_fn, num_samples, dt, epsilon, x_0):

    def ula_step(carry, x):
        key, param = carry
        key, param = ula_kernel(key, param, score_fn, dt, epsilon)
        return (key, param), param

    carry = (rng, x_0)
    _, samples = lax.scan(ula_step, carry, None, num_samples)
    return samples[-1]

def langevin_sample(num_iterations, epsilon, step_size, xy_init: jnp.ndarray, rng_0, score_fn):
    return ula_sampler_full_jax_jit(rng_0, score_fn, num_iterations, step_size, epsilon, xy_init)

# def langevin_sample(num_iterations, epsilon, step_size, xy_init: jnp.ndarray, rng_0, score_fn):
#     xy = xy_init
#     rngs = random.split(rng_0, num_iterations)
#     for rng in rngs:
#         brownian_motion = random.normal(rng, (xy.shape[-1],)) 
#         xy = xy - step_size * score_fn(xy) + jnp.sqrt(2.*epsilon)* brownian_motion * jnp.sqrt(step_size)
#     return xy


langevin_sample = jax.vmap(langevin_sample, in_axes=[None, None, None, 0, 0, None])
# langevin_sample = jax.jit(langevin_sample, static_argnums=[0,5])

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
