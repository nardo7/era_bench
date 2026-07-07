from argparse import Namespace
from typing import cast

from isaaclab_tasks.utils import parse_env_cfg
import os
from era_bench.tasks.manager_based.common.recorders import RecorderManagerCfg
from era_bench.tasks.manager_based.mirror.era_bench_env_cfg import EraBenchEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner, CuroboPlannerCfg
from era_bench.curobo_planner_extended import ExtendedCuroboPlanner
from isaaclab.assets.articulation import Articulation
from era_bench.tasks.manager_based.arc_embodied.scene_cfg import FRANKA_GRASP_CENTER_OFFSET


def configure_env(args_cli: Namespace) -> ManagerBasedRLEnvCfg:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    env_cfg: EraBenchEnvCfg = cast(
        EraBenchEnvCfg, parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    )
    # if not isinstance(env_cfg, EraBenchEnvCfg):
    #     raise ValueError(f"Expected env_cfg to be of type EraBenchEnvCfg, but got {type(env_cfg)}")

    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Important: do not concatenate observations if you want recorder to store structured obs.
    env_cfg.observations.policy.concatenate_terms = False

    # Attach the recorder: records actions/states/obs + generates per-episode MP4 videos.
    env_cfg.recorders = (
        RecorderManagerCfg(sensor_name="table_cam") if args_cli.store_videos else ActionStateRecorderManagerCfg()
    )
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES

    # Keep timeout termination so episodes reset properly
    # (remove this line if you want infinite-length demos)
    # if hasattr(env_cfg.terminations, "time_out"):
    #     env_cfg.terminations.time_out = None

    return env_cfg


def configure_curobo_planner(env: ManagerBasedRLEnv, task_name: str, env_id: int) -> ExtendedCuroboPlanner:
    """Configure cuRobo planner, optionally with extended goalset support.

    Args:
        env: The Isaac Lab environment
        task_name: Task name for planner config auto-detection
        env_id: Environment ID

    Returns:
        ExtendedCuroboPlanner instance
    """
    robot: Articulation = env.scene["robot"]
    planner_cfg = CuroboPlannerCfg.from_task_name(task_name)
    planner_cfg.visualize_plan = False
    planner_cfg.visualize_spheres = False
    planner_cfg.debug_planner = False
    planner_cfg.retreat_distance = 0.0
    planner_cfg.approach_distance = 0.0
    planner_cfg.time_dilation_factor = 1.0
    planner_cfg.interpolation_dt = 0.06
    planner_cfg.trajopt_tsteps = 40
    planner_cfg.n_repeat = 10

    # planner_cfg.debug_planner = False

    planner = ExtendedCuroboPlanner(
        env=env,
        robot=robot,
        config=planner_cfg,
        env_id=env_id,
        grasp_center_offset=FRANKA_GRASP_CENTER_OFFSET,
    )

    return planner
