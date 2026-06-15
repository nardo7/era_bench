from argparse import Namespace
from typing import cast

from isaaclab_tasks.utils import parse_env_cfg
import os
from era_bench.tasks.manager_based.mirror.mdp import RecorderManagerCfg
from era_bench.tasks.manager_based.mirror.era_bench_env_cfg import EraBenchEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode


def configure_env(args_cli: Namespace) -> ManagerBasedRLEnvCfg:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    env_cfg: EraBenchEnvCfg = cast(
        EraBenchEnvCfg, parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    )
    if not isinstance(env_cfg, EraBenchEnvCfg):
        raise ValueError(f"Expected env_cfg to be of type EraBenchEnvCfg, but got {type(env_cfg)}")

    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Important: do not concatenate observations if you want recorder to store structured obs.
    env_cfg.observations.policy.concatenate_terms = False

    # Attach the recorder: records actions/states/obs + generates per-episode MP4 videos.
    env_cfg.recorders = RecorderManagerCfg() if args_cli.store_videos else ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES

    # Keep timeout termination so episodes reset properly
    # (remove this line if you want infinite-length demos)
    # if hasattr(env_cfg.terminations, "time_out"):
    #     env_cfg.terminations.time_out = None

    return env_cfg
