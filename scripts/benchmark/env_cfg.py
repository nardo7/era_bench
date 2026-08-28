from argparse import Namespace
from typing import cast

from era_bench.tasks.manager_based.common.dataset_handler import AppendHDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
import os
from era_bench.tasks.manager_based.arc_embodied.env.env_cfg import EraBenchEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.datasets.hdf5_dataset_file_handler import HDF5DatasetFileHandler
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner, CuroboPlannerCfg
from era_bench.curobo_planner_extended import ExtendedCuroboPlanner
from isaaclab.assets.articulation import Articulation
from era_bench.tasks.manager_based.arc_embodied.env.scene_cfg import FRANKA_GRASP_CENTER_OFFSET


def configure_env(args_cli: Namespace) -> ManagerBasedRLEnvCfg:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    env_cfg: EraBenchEnvCfg = cast(
        EraBenchEnvCfg, parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    )
    # if not isinstance(env_cfg, EraBenchEnvCfg):
    #     raise ValueError(f"Expected env_cfg to be of type EraBenchEnvCfg, but got {type(env_cfg)}")

    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Keep policy and camera observations structured so recorder terms store named obs/<term> datasets.
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.observations.camera.concatenate_terms = False

    # Attach the recorder: records actions/states/obs + generates per-episode MP4 videos.
    env_cfg.recorders = env_cfg.recorders
    env_cfg.recorders.set_store_videos(args_cli.store_videos)

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    env_cfg.recorders.dataset_file_handler_class_type = (
        HDF5DatasetFileHandler if args_cli.overwrite_dataset else AppendHDF5DatasetFileHandler
    )

    return env_cfg


def configure_curobo_planner(
    env: ManagerBasedRLEnv, task_name: str, env_id: int, debug: bool = False
) -> ExtendedCuroboPlanner:
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
    planner_cfg.debug_planner = debug
    planner_cfg.retreat_distance = 0.0
    planner_cfg.approach_distance = 0.0
    planner_cfg.time_dilation_factor = 1.0
    planner_cfg.interpolation_dt = 0.06
    planner_cfg.trajopt_tsteps = 60
    planner_cfg.n_repeat = 10
    planner_cfg.enable_graph = True

    # Input-grid cubes are visual references. They deliberately have no PhysX
    # colliders, and must also be excluded from CuRobo's USD-stage extraction.
    # ``world_ignore_substrings`` replaces the planner's defaults when set.
    env_prim_path = f"/World/envs/env_{env_id}"
    planner_cfg.world_ignore_substrings = [
        *(
            planner_cfg.world_ignore_substrings
            or [
                f"{env_prim_path}/Robot",
                f"{env_prim_path}/target",
                "/World/defaultGroundPlane",
                "/curobo",
            ]
        ),
        f"{env_prim_path}/input_grid_cube_",
    ]

    # planner_cfg.debug_planner = False

    planner = ExtendedCuroboPlanner(
        env=env,
        robot=robot,
        config=planner_cfg,
        env_id=env_id,
        grasp_center_offset=FRANKA_GRASP_CENTER_OFFSET,
        use_native_goalset=False,
    )

    return planner
