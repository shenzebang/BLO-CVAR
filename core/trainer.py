import jax
import jax.numpy as jnp
import jax.random as random
import optax
import wandb

class JaxTrainer:
    def __init__(self,
                 cfg,
                 method,
                 rng):
        self.cfg = cfg
        self.method = method
        self.rng = rng
    
    def fit(self, ):
        def _value_and_grad_fn(params, rng):
            return self.method.value_and_grad_fn(params, rng)
        
        if self.cfg.backend.use_pmap_train and jax.local_device_count() > 1:
            _value_and_grad_fn = jax.pmap(_value_and_grad_fn, in_axes=(None, None, 0))

            def value_and_grad_fn(params, rng):

                rngs = random.split(rng, jax.local_device_count())
                # compute in parallel
                v_g_etc = _value_and_grad_fn(params, rngs)
                v_g_etc = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), v_g_etc)
                return v_g_etc
        else:
            value_and_grad_fn = jax.jit(_value_and_grad_fn)

        @jax.jit
        def step_fn(params, opt_state, grad, scale=1):
            updates, opt_state = self.optimizer.update(grad, opt_state, params)   
            updates = jax.tree_util.tree_map(lambda g: scale * g, updates)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        
        def plot_fn(params, rng):
            return self.method.plot_fn(params, rng)
        
        @jax.jit
        def metric_fn(params, rng):
            return self.method.metric_fn(params, rng)
        
        def test_metric_and_log_fn(params, rng):
            metrics = metric_fn(params, rng)
            wandb.log(metrics)
        
        rng_metric_0, rng_0 = jax.random.split(self.rng)
        wandb.define_metric("metric/step")
        wandb.define_metric("metric/*", step_metric="metric/step")
        if self.cfg.pde_instance.test_metric:
            test_metric_and_log_fn(self.params, rng_metric_0)

        opt_state = self.optimizer.init(self.params)
        rngs = random.split(rng_0, self.cfg.train.total_iterations)
        for rng in rngs:
            rng_train, rng_test, rng_plot = random.split(rng, 3)
            v_g_etc = value_and_grad_fn(self.params, rng_train)
            self.params, opt_state = step_fn(self.params, opt_state, v_g_etc["grad"])

            v_g_etc.pop("grad")
            # TODO: implement test

            wandb.log(v_g_etc)

            # TODO: implement plot
            plot_fn(self.params, rng_plot) 