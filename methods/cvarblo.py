from jax.random import KeyArray
from omegaconf import DictConfig
from api import BLOProblem, BLOSolver, TrainState
import jax, optax
from utils.sampling import LangevinSampler
import jax.numpy as jnp

# smoothed version of max(x, 0)


class CVaRBLOSolver(BLOSolver):
    def __init__(self, blo_problem: BLOProblem, cfg: DictConfig, rng: KeyArray) -> None:
        super().__init__(blo_problem, cfg, rng)
        if blo_problem.lower_level_problem.min_or_max != 'max':
            raise ValueError("CVaR BLO solver supports only pessimistic minimal selection.")
        self.upper_temperature = 1.
        self.lower_temperature = 1.
        self.quantile = .95
        self.delta = .1
        self.gamma = .01
        self.upper_level_optimizer = optax.sgd(learning_rate=.01, momentum=.9)
        self.initial_distribution_lower_level = [] # TODO
        self.lower_level_langevin = LangevinSampler(blo_problem.lower_level_problem, self.initial_distribution_lower_level, cfg)
        self.problem_dimension = 2
        

    def init_train_state(self, rng_init):
        param_init = jax.random.normal(rng_init, [self.problem_dimension])
        return TrainState(
            opt_state=self.upper_level_optimizer.init(param_init),
            UL_param=param_init,
            LL_param=None,
        )

    def grad_upper_level_fn(self, iter_upper_level, samples_lower_level):
        # compute beta
        _, beta=self.CVAR_delta(iter_upper_level, samples_lower_level)

        # E[\partial_\theta g]
        partial_theta_g_fn = jax.grad(self.blo_problem.lower_level_problem.value_fn, argnums=0)
        partial_theta_g_vmap_x_fn = jax.vmap(partial_theta_g_fn, in_axes=[None, 0])
        E_partial_theta_g = jnp.mean(partial_theta_g_vmap_x_fn(iter_upper_level, samples_lower_level), axis=0)

        
        def grad_upper_level_single_fn(sample_lower_level):
            hf_fn = lambda theta, x: self.h(self.blo_problem.upper_level_problem.value_fn(theta, x) - beta)
            vg_theta_hf_fn = jax.value_and_grad(hf_fn, argnums=0)
            hf, d_theta_hf = vg_theta_hf_fn(iter_upper_level, sample_lower_level)
            partial_theta_log_pi_lambda_theta_x = (E_partial_theta_g - partial_theta_g_fn(iter_upper_level, sample_lower_level))/self.lower_temperature
            return (d_theta_hf + partial_theta_log_pi_lambda_theta_x * hf)/self.delta
        
        grad_upper_level_single_fn = jax.vmap(grad_upper_level_single_fn, in_axes=[0,])

        return jnp.mean(grad_upper_level_single_fn(samples_lower_level), axis=0)

    def opt_step_fn(self, params, opt_state, grad):
        updates, opt_state = self.upper_level_optimizer.update(grad, opt_state, params)   
        params = optax.apply_updates(params, updates)
        return params, opt_state
    

    def step_fn(self, train_state: TrainState, rng):
        iter_upper_level = train_state.UL_param
        opt_state_upper_level = train_state.opt_state
        samples_lower_level = self.lower_level_langevin.sample_fn(rng)
        grad_upper_level = self.grad_upper_level_fn(iter_upper_level, samples_lower_level)
        iter_upper_level, opt_state_upper_level = self.opt_step_fn(iter_upper_level, opt_state_upper_level, grad_upper_level)
        return TrainState(
            opt_state=opt_state_upper_level,
            UL_param=iter_upper_level, 
            LL_param=None # CVaR does not return lower-level iterate
            )
    
    def CVAR_delta(self, iter_upper_level, samples_lower_level):
        f = self.blo_problem.upper_level_problem.value_fn
        beta2=0
        h1_sum=0
        sample_f_values=[]
        for Y_sample in samples_lower_level:
            f_value = f(iter_upper_level, Y_sample)
            sample_f_values.append(f_value)
        #print('sample_f_values:',sample_f_values)
        beta2=jnp.nanquantile(sample_f_values, 1-self.delta)    
        # print('beta2', beta2)
        for Y_sample in samples_lower_level:
            f_value = f(iter_upper_level, Y_sample)
            h1=self.h(f_value - beta2, self.gamma)
            h1_sum+=h1
        expected_h=h1_sum/(self.delta*len(samples_lower_level))
        # print('expected_h:', expected_h) 
        return beta2 + expected_h, beta2 
    
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
        