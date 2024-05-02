from omegaconf import DictConfig
from api import BLOProblem, BLOSolver, TrainState
import jax, optax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from jax import lax
from utils.common import optimizer_step_fn

class IAPTT_GM(BLOSolver):
    def __init__(self, blo_problem: BLOProblem, cfg: DictConfig, rng) -> None:
        super().__init__(blo_problem, cfg, rng)
        if blo_problem.problem_UL.min_or_max != 'max':
            raise ValueError("Currently, IAPTT-GM BLO solver supports only pessimistic minimal selection.")
        self.optimizer_UL = optax.sgd(learning_rate=cfg.solver.upper_level_optimizer.step_size, 
                                               momentum=cfg.solver.upper_level_optimizer.momentum)
        self.optimizer_LL = optax.sgd(learning_rate=cfg.solver.lower_level_optimizer.step_size, 
                                               momentum=cfg.solver.lower_level_optimizer.momentum)
        self.optimizer_auxiliary = optax.sgd(learning_rate=cfg.solver.auxiliary_optimizer.step_size, 
                                               momentum=cfg.solver.auxiliary_optimizer.momentum)
        
    def init_train_state(self, rng_init):
        # TODO: implement the params initialization based on problem
        rng_init_UL, rng_init_auxiliary = random.split(rng_init, 2)
        param_init_UL = jax.random.normal(rng_init_UL, [self.blo_problem.dim.UL_dim])
        param_init_auxiliary = jax.random.normal(rng_init_auxiliary, [self.blo_problem.dim.LL_dim]) * 20. - 10.
        return TrainState(
            opt_state={
                "UL": self.optimizer_UL.init(param_init_UL),
                "auxiliary": self.optimizer_auxiliary.init(param_init_auxiliary)},
            UL_param=param_init_UL,
            LL_param=None,
            auxiliary=param_init_auxiliary, # initializer for LL
        )
    
    def step_fn(self, ts: TrainState, rng):
        rng_LL, rng_UL, rng_auxiliary = random.split(rng, 3)
        grad_zx_LL_fn = jax.grad(self.solve_LL_problem_fn, argnums=[0, 1], has_aux=True)
        (grad_z, grad_x), y_bar_k = grad_zx_LL_fn(ts.auxiliary, ts.UL_param, rng_LL)
        # UL update
        x, opt_state_UL = optimizer_step_fn(self.optimizer_UL, ts.UL_param, grad_x, ts.opt_state["UL"], rng_UL)
        # auxiliary update
        z, opt_state_auxiliary = optimizer_step_fn(self.optimizer_auxiliary, ts.auxiliary, grad_z, ts.opt_state["auxiliary"], rng_auxiliary)
        return TrainState(
            opt_state={
                "UL": opt_state_UL,
                "auxiliary": opt_state_auxiliary,
            },
            UL_param=x,
            LL_param=y_bar_k,
            auxiliary=z,
        ), {"param_UL": x, "param_LL": y_bar_k, "grad_z": grad_z, "grad_x": grad_x}

    def solve_LL_problem_fn(self, param_LL_init, param_UL, rng): # output the F(x, y_{\bar k}), y_{\bar k}
        grad_y_fn = jax.grad(self.blo_problem.problem_LL.value_fn, argnums=1)
        # solve LL 
        def _step_fn(y, _x):
            y = y - self.cfg.solver.lower_level_optimizer.step_size * grad_y_fn(param_UL, y)
            return y, y
        _, y_trajectory = lax.scan(_step_fn, param_LL_init, None, 
                            self.cfg.solver.lower_level_optimizer.number_of_steps)
        # pessimistic trajectory truncation
        UL_value_fn_vmap = jax.vmap(self.blo_problem.problem_UL.value_fn, in_axes=[None, 0])
        F_values = UL_value_fn_vmap(param_UL, y_trajectory)
        k = jnp.argmax(F_values)
        # return F(x, y_{\bar k}), y_{\bar k}
        return F_values[k], y_trajectory[k]
    
    def plot_fn(self, rng, ts: TrainState):
        pass