import jax.numpy as np
import jax
import jax.random as random
import matplotlib.pyplot as plt

def g(xy: np.ndarray):
    assert xy.ndim == 1
    x, y = xy[0], xy[1]
    conditions = [
        np.logical_and(y > np.sin(x), np.logical_and(0 < x,  x < 3*np.pi)),
        np.logical_and(x < 0, y > 0),
        np.logical_and(np.logical_and(-3*np.pi < y, y < 0), x < np.sin(y)),
        np.logical_and(y < -3*np.pi, x < 0),
        np.logical_and(y < -np.sin(x) - 3*np.pi, np.logical_and(0 < x, x <3*np.pi)),
        np.logical_and(y < -3*np.pi, x > 3*np.pi),
        np.logical_and(np.logical_and(-3*np.pi < y, y < 0), x > -np.sin(y) + 3*np.pi),
        np.logical_and(y > 0, x > 3*np.pi),
    ]
    functions = [
        (y - np.sin(x))**2,
        (y - x)**2,
        (x - np.sin(y))**2,
        (y + x + 3*np.pi)**2,
        (y + np.sin(x) + 3*np.pi)**2,
        (y - x + 6*np.pi)**2,
        (x + np.sin(y) - 3*np.pi)**2,
        (x - 3*np.pi + y)**2,
        np.zeros([])
    ]
    return np.piecewise(np.ones([]), conditions, functions)

def g_grad(xy: np.ndarray):
    assert xy.ndim == 1
    x, y = xy[0], xy[1]
    conditions = [
        np.logical_and(y > np.sin(x), np.logical_and(0 < x,  x < 3*np.pi)),
        np.logical_and(x < 0, y > 0),
        np.logical_and(np.logical_and(-3*np.pi < y, y < 0), x < np.sin(y)),
        np.logical_and(y < -3*np.pi, x < 0),
        np.logical_and(y < -np.sin(x) - 3*np.pi, np.logical_and(0 < x, x <3*np.pi)),
        np.logical_and(y < -3*np.pi, x > 3*np.pi),
        np.logical_and(np.logical_and(-3*np.pi < y, y < 0), x > -np.sin(y) + 3*np.pi),
        np.logical_and(y > 0, x > 3*np.pi),
    ]
    functions = [
        np.array([-2 * (y - np.sin(x)) * np.cos(x), 2 * (y - np.sin(x))]),
        np.array([-2 * (y - x), 2 * (y - x)]),
        np.array([2 * (x - np.sin(y)), -2 * np.cos(y) * (x - np.sin(y))]),
        np.array([2 * (y + x + 3*np.pi), 2 * (y + x + 3*np.pi)]),
        np.array([2 * (y + np.sin(x) + 3*np.pi) * np.cos(x), 2 * (y + np.sin(x) + 3*np.pi)]),
        np.array([-2 * (y - x + 6*np.pi), 2 * (y - x + 6*np.pi)]),
        np.array([2 * (x + np.sin(y) - 3*np.pi), 2 * np.cos(y) * (x + np.sin(y) - 3*np.pi)]),
        np.array([2 * (x - 3*np.pi + y), 2 * (x - 3*np.pi + y)]),
        np.array([0, 0])
    ]
    return np.piecewise(np.ones([2]), conditions, functions)


nabla_g = jax.jit(jax.grad(g))

rng = random.PRNGKey(1)
rng_init, rng_langevin = random.split(rng, 2)

n_particle = 2000

xy_inits = random.uniform(rng_init, [n_particle, 2]) * 20. - 10.
rngs_langevin = random.split(rng_langevin, n_particle)
num_iterations = 10000
epsilon = 1e-3
step_size = 1

# @jax.jit
def langevin_sample(num_iterations, epsilon, step_size, xy_init, rng_0):
    xy = xy_init
    rngs = random.split(rng_0, num_iterations)
    for rng in rngs:
        brownian_motion = random.normal(rng, (2,)) 
        xy = xy - step_size * nabla_g(xy) + np.sqrt(np.ones([])*2*epsilon)* brownian_motion * np.sqrt(np.ones([])*step_size)
    return xy
# langevin_sample = jax.jit(langevin_sample, static_argnums=[0])

langevin_sample = jax.vmap(langevin_sample, in_axes=[None, None, None, 0, 0])
xys = langevin_sample(num_iterations, epsilon, step_size, xy_inits, rngs_langevin)

plt.scatter(xys[:, 0], xys[:, 1], color='blue')
plt.title("Samples at Iteration 100 (Epsilon = 0.01)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

