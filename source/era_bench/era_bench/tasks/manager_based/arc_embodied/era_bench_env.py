from isaaclab.envs import ManagerBasedRLMimicEnv
from .era_bench_env_cfg import EraBenchEnvCfg


class EraBenchEnv(ManagerBasedRLMimicEnv):
    """Environment for the ERA Bench task."""

    def __init__(self, cfg: EraBenchEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
