import jax
import jax.numpy as jnp
import jax.random as random
import wandb
from api import BLOSolver, BLOProblem
from tqdm import tqdm

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
        # self.solver.plot_fn(self.rng, self.train_state)

        step_fn = jax.jit(self.solver.step_fn)
        # step_fn = self.solver.step_fn

        rngs = random.split(self.rng, self.cfg.train.total_iterations)
        for rng in tqdm(rngs):
            rng_train, _, rng_plot = random.split(rng, 3)
            self.train_state, stats = step_fn(self.train_state, rng_train)
            # test
            # log
            for stat in stats:
                print(stat, stats[stat])
            # plot (only possible for 2D problems)
            # if self.problem.dim.LL_dim == 2:
            #     self.solver.plot_fn(rng_plot, self.train_state)
            # save snapshot
        
        # test final model
        # log final statistics
        # save model