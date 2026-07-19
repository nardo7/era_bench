from torch import Tensor
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from .era_bench_env_cfg import EraBenchEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

from .mdp.commands import MAX_OUTPUT_OBJECTS_PER_GRID
from .success_positions import ReserveCubeStackBuffer, SuccessPositionBuffer


class EraBenchEnv(ManagerBasedRLMimicEnv):
    """Environment for the ERA Bench task."""

    cfg: EraBenchEnvCfg
    success_positions: SuccessPositionBuffer
    reserve_cube_stacks: ReserveCubeStackBuffer
    reserve_cube_was_grasped: Tensor
    success_positions_reached: Tensor

    def __init__(self, cfg: EraBenchEnvCfg, **kwargs):
        self.reserve_cube_was_grasped = torch.zeros(
            (cfg.scene.num_envs, len(cfg.scene.reserve_cubes)), dtype=torch.bool, device=cfg.sim.device
        )
        self.success_positions_reached = torch.zeros(
            (cfg.scene.num_envs, MAX_OUTPUT_OBJECTS_PER_GRID), dtype=torch.bool, device=cfg.sim.device
        )
        super().__init__(cfg, **kwargs)
        self.success_positions = SuccessPositionBuffer(
            num_envs=self.num_envs,
            max_targets=MAX_OUTPUT_OBJECTS_PER_GRID,
            device=self.device,
        )
        self.reserve_cube_stacks = ReserveCubeStackBuffer(
            num_envs=self.num_envs,
            num_stacks=cfg.scene.reserve_stacks_amount,
            cubes_per_stack=cfg.scene.reserve_cubes_amount_per_stack,
            device=self.device,
        )

    def load_managers(self):
        super().load_managers()
        load_task_command = self.command_manager.get_term("load_task")
        load_task_command._initialize_from_dataset()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.reserve_cube_was_grasped[env_ids] = False
        self.success_positions_reached[env_ids] = False

    def get_reserve_cubes(self) -> list[RigidObject]:
        """Get the reserve cubes."""
        reserve_cube_names = self.cfg.scene.reserve_cubes
        entities = [SceneEntityCfg(name) for name in reserve_cube_names]
        return [self.scene[entity.name] for entity in entities]

    def get_live_reserve_cube_stack_poses(self) -> Tensor:
        """Return live reserve poses in physical stack order, not grouped by color."""
        scene_cfg = self.cfg.scene
        poses = []
        for stack_index in range(scene_cfg.reserve_stacks_amount):
            cubes = [
                self.scene[scene_cfg.get_reserve_cube_name(stack_index, cube_index)]
                for cube_index in range(scene_cfg.reserve_cubes_amount_per_stack)
            ]
            poses.append(
                torch.stack(
                    [torch.cat((cube.data.root_link_pos_w, cube.data.root_quat_w), dim=-1) for cube in cubes], dim=1
                )
            )
        return torch.stack(poses, dim=1)
