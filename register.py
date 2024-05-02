from omegaconf import DictConfig
from example_problems import handcraft
from methods import cvarblo, iaptt_gm

BLOINSTANCES = {
    "handcraft": handcraft.HandCraft,
}

BLOSOLVERS = {
    "cvar-langevin": cvarblo.CVaRBLOSolver,
    "iaptt-gm": iaptt_gm.IAPTT_GM,  
}

def get_BLO_instance(cfg: DictConfig):
    return BLOINSTANCES[cfg.problem.name]
    

def get_method(cfg: DictConfig):
    return BLOSOLVERS[cfg.solver.name]