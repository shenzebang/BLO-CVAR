import optax
from optax._src.base import GradientTransformation

def optimizer_step_fn(optimizer: GradientTransformation, params, grad, opt_state, rng):
    updates, opt_state = optimizer.update(grad, opt_state, params)   
    params = optax.apply_updates(params, updates)
    return params, opt_state