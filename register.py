from omegaconf import DictConfig
from example_problems import handcraft
from methods import cvarblo

BLOINSTANCES = {
    "handcraft": handcraft.HandCraft,
}

BLOSOLVERS = {
    "cvar": cvarblo.CVaRBLOSolver
}

def get_BLO_instance(cfg: DictConfig):
    return BLOINSTANCES[cfg.problem.name]
    

def get_method(cfg: DictConfig):
    return BLOSOLVERS[cfg.solver.name]