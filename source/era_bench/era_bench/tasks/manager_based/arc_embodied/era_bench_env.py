from torch import Tensor
import torch

from era_bench.tasks.manager_based.arc_embodied.config import Colors
from isaaclab.envs import ManagerBasedRLMimicEnv
from .era_bench_env_cfg import EraBenchEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject


class EraBenchEnv(ManagerBasedRLMimicEnv):
    """Environment for the ERA Bench task."""

    cfg: EraBenchEnvCfg
    success_positions: dict[Colors, list[Tensor]] = {}

    def __init__(self, cfg: EraBenchEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def get_reserve_cubes(self) -> list[RigidObject]:
        """Get the reserve cubes."""
        reserve_cube_names = self.cfg.scene.reserve_cubes
        entities = [SceneEntityCfg(name) for name in reserve_cube_names]
        return [self.scene[entity.name] for entity in entities]

    def get_reserve_cubes_by_color(self) -> dict[Colors, list[RigidObject]]:
        """Get the reserve cubes grouped by color."""
        reserve_cubes = self.get_reserve_cubes()
        cubes_by_color: dict[Colors, list[RigidObject]] = {}
        for cube in reserve_cubes:
            color = self.cfg.scene.get_reserve_cube_color(cube.body_names[0])
            if color not in cubes_by_color:
                cubes_by_color[color] = []
            cubes_by_color[color].append(cube)
        return cubes_by_color

    def get_reserve_cubes_positions(self) -> list[Tensor]:
        """Get the positions of the reserve cubes."""
        reserve_cubes = self.get_reserve_cubes()
        return [cube.data.root_link_pos_w for cube in reserve_cubes]

    def get_reserve_cubes_positions_by_color(self) -> dict[Colors, Tensor]:
        """Get the positions of the reserve cubes grouped by color."""
        reserve_cubes_by_color = self.get_reserve_cubes_by_color()
        positions_by_color: dict[Colors, Tensor] = {}
        for color, cubes in reserve_cubes_by_color.items():
            positions_by_color[color] = torch.stack(
                [cube.data.root_link_pos_w for cube in cubes],
                dim=1 if len(cubes[0].data.root_link_pos_w.shape) == 3 else 0,
            )
        return positions_by_color
