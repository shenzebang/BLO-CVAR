import jax.numpy as np
import jax


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
    return np.piecewise(np.zeros([]), conditions, functions)

xy = np.array([100., 80.])
nabla_g = jax.jit(jax.grad(g))
print(nabla_g(xy))
