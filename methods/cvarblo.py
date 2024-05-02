from omegaconf import DictConfig
from api import BLOProblem, BLOSolver, TrainState, Distribution
import jax, optax
from utils.sampling import BLOLangevinSampler
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from jax import lax


class InitialDistribution(Distribution):
    def __init__(self, dim) -> None:
        self.dim = dim

    def sample(self, rng, batch_size):
        return random.uniform(rng, [batch_size, self.dim]) * 20. - 10.
            
class CVaRBLOSolver(BLOSolver):
    def __init__(self, blo_problem: BLOProblem, cfg: DictConfig, rng) -> None:
        super().__init__(blo_problem, cfg, rng)
        # TODO: should be either min-max or max-max since there are two variables.
        if blo_problem.problem_UL.min_or_max != 'max':
            raise ValueError("Currently, CVaR BLO solver supports only pessimistic minimal selection.")
        # Hyperparameters 
        self.lower_temperature = cfg.solver.langevin_sampler.temperature
        self.delta = cfg.solver.cvar.delta # (1-delta) is the quantile
        self.gamma = cfg.solver.cvar.gamma # parameter for the smoothed version of max{0, x}

        # TODO: choose optimizer based on cfg
        self.optimizer_UL = optax.sgd(learning_rate=cfg.solver.upper_level_optimizer.step_size, 
                                               momentum=cfg.solver.upper_level_optimizer.momentum)
        self.initial_distribution_lower_level = self._get_initial_distribution_LL()
        
        self.langevin_LL = BLOLangevinSampler(blo_problem.problem_LL, self.initial_distribution_lower_level, cfg)
        

    def init_train_state(self, rng_init):
        # TODO: implement the param initialization based on problem
        param_init = jax.random.normal(rng_init, [self.blo_problem.dim.UL_dim])
        return TrainState(
            opt_state=self.optimizer_UL.init(param_init),
            UL_param=param_init,
            LL_param=None,
            auxiliary=None
        )

    def step_fn(self, ts: TrainState, rng):
        param_UL, opt_state_UL = ts.UL_param, ts.opt_state
        
        samples_LL = self.langevin_LL.sample_fn(rng, param_UL)
        grad_UL, stats = self.grad_UL_fn(param_UL, samples_LL)
        # need to negate the grad_UL since we are maximizing the UL objective
        # grad_UL = jax.tree_map(lambda x: -x, grad_UL)
        param_UL, opt_state_UL = self.opt_step_fn(param_UL, opt_state_UL, grad_UL)
        stats["param_UL"] = param_UL
        return TrainState(
            opt_state=opt_state_UL,
            UL_param=param_UL, 
            LL_param=None,
            auxiliary=None # CVaR does not return lower-level iterate
            ), stats
    
    def grad_UL_fn(self, param_UL, samples_LL):
        # compute beta
        CVaR, beta = self.CVaR_and_beta_fn(param_UL, samples_LL)
        # E[\partial_\theta g]
        partial_theta_g_fn = jax.grad(self.blo_problem.problem_LL.value_fn, argnums=0)
        partial_theta_g_vmap_x_fn = jax.vmap(partial_theta_g_fn, in_axes=[None, 0])
        E_partial_theta_g = jnp.mean(partial_theta_g_vmap_x_fn(param_UL, samples_LL), axis=0)

        
        def grad_upper_level_single_fn(sample_lower_level):
            hf_fn = lambda theta, x: self.h(self.blo_problem.problem_UL.value_fn(theta, x) - beta)
            vg_theta_hf_fn = jax.value_and_grad(hf_fn, argnums=0)
            hf, d_theta_hf = vg_theta_hf_fn(param_UL, sample_lower_level)
            partial_theta_log_pi_lambda_theta_x = (E_partial_theta_g - partial_theta_g_fn(param_UL, sample_lower_level))/self.lower_temperature
            return (d_theta_hf + partial_theta_log_pi_lambda_theta_x * hf)/self.delta
        
        grad_upper_level_single_fn = jax.vmap(grad_upper_level_single_fn, in_axes=[0,])

        return jnp.mean(grad_upper_level_single_fn(samples_LL), axis=0), {"CVaR": CVaR, "norm of mean_LL": jnp.sum(jnp.mean(samples_LL, axis=0) ** 2)}
    
    def CVaR_and_beta_fn(self, param_UL, samples_LL):
        f_vmap_x = jax.vmap(self.blo_problem.problem_UL.value_fn, in_axes=[None, 0])
        sample_f_values = f_vmap_x(param_UL, samples_LL)
        
        # initialize beta with (1-delta) quantile
        beta_init = jnp.nanquantile(sample_f_values, 1. - self.delta, axis=0)
        # update beta with GD
        beta = self.update_beta(beta_init, sample_f_values)
        # Compute the smoothed CVaR using the optimal beta
        CVaR = self.beta_sample_to_CVaR_fn(beta, sample_f_values)
        return CVaR, beta 
    
    def beta_sample_to_CVaR_fn(self, beta, sample_f_values):
        h_vmap_fx = jax.vmap(self.h)
        return jnp.mean(h_vmap_fx(sample_f_values - beta)) / self.delta + beta
    
    def update_beta(self, beta_init, samples):
        def gd_step(beta, x):
            grad_beta_fn = jax.grad(self.beta_sample_to_CVaR_fn, argnums=0)
            beta = beta - self.cfg.solver.beta_optimizer.step_size * grad_beta_fn(beta, samples)
            return beta, None
        
        beta, _ = lax.scan(gd_step, beta_init, None, self.cfg.solver.beta_optimizer.number_of_steps)
        return beta

    
    # smoothed version of max(x, 0)
    def h(self, fx: jnp.ndarray):
        assert fx.ndim == 0
        conditions = [
            fx > self.gamma,
            jnp.logical_and(fx >= -self.gamma, fx <= self.gamma),
        ]
        functions = [
            fx,
            fx ** 2. / (4. * self.gamma) + fx /2. + self.gamma/4.,
            jnp.zeros([])
        ]
        return jnp.piecewise(jnp.zeros([]), conditions, functions)
    
    def opt_step_fn(self, params, opt_state, grad):
        updates, opt_state = self.optimizer_UL.update(grad, opt_state, params)   
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def _get_initial_distribution_LL(self,) -> Distribution:
        return InitialDistribution(self.blo_problem.problem_LL.problem_dimension.LL_dim)

    def plot_fn(self, rng, ts: TrainState):
        if self.blo_problem.dim.LL_dim == 2:
            samples_LL = self.langevin_LL.sample_fn(rng, ts.UL_param)
            plt.scatter(samples_LL[:, 0], samples_LL[:, 1], color='blue')
            plt.title("Lower-level Samples")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()
        else:
            raise ValueError("plot_fn only supports 2D problems!")
