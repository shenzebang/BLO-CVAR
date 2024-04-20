import hydra
import wandb
from omegaconf import OmegaConf
import jax.random as random
from register import *
from core.trainer import JaxTrainer

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project=f"BLO-{cfg.info.algorithm_name}",
        # Track hyperparameters and run metadata
        config=OmegaConf.to_container(cfg),
        # hyperparameter tuning mode or normal mode.
        # name=cfg.mode
    )
    rng = random.PRNGKey(cfg.seed)

    # create problem instance
    BLO_instance = get_BLO_instance(cfg)

    # create method instance
    method = get_method(cfg)

    # Construct the JaxTrainer
    trainer = JaxTrainer(cfg=cfg, method=method, rng=rng)

    # Fit the model
    trainer.fit()

    # Test the model
    wandb.finish()

if __name__ == '__main__':
    main()