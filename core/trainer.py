import jax
import jax.numpy as jnp
import jax.random as random
import wandb
from api import BLOSolver, BLOProblem

class JaxTrainer:
    def __init__(self,
                 cfg,
                 solver: BLOSolver,
                 problem: BLOProblem,
                 rng):
        self.cfg = cfg
        self.solver = solver
        self.problem = problem
        self.rng, rng_init = random.split(rng, 2)
        self.train_state = solver.init_train_state(rng_init)
    
    def fit(self, ):
        '''
            Handle test, log, and backend configuration.
        '''

        rngs = random.split(self.rng, self.cfg.train.total_iterations)
        for rng in rngs:
            rng_train, _, _ = random.split(rng, 3)
            self.train_state = self.solver.step_fn(self.train_state, rng_train)