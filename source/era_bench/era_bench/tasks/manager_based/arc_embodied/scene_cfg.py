from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from .config import DEFAULT_COLORS, Colors
import torch


@configclass
class EraBenchSceneCfg(InteractiveSceneCfg):

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    )

    table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=256,
        width=384,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.8, 0.0, 1.7), rot=(0.24184, -0.66446, -0.66446, 0.24184), convention="ros"),
    )

    origin_top_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/origin_top_cam",
        update_period=0.0,
        height=384,
        width=384,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 2.4), rot=(0.0, 0.70711, 0.70711, 0.0), convention="ros"),
    )

    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING  # pyright: ignore[reportAssignmentType]

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.2, 0, -0.05),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    @property
    def gripper_joint_names(self) -> list[str]:
        return ["panda_finger_.*"]

    @property
    def gripper_open_val(self) -> float:
        return 0.04

    @property
    def gripper_threshold(self) -> float:
        return 0.005

    @property
    def line_name(self) -> str:
        return "line"

    @property
    def object_names(self) -> list[str]:
        """Names of cube objects currently registered in this scene config."""
        return sorted(k for k in self.__dict__ if k.startswith("cube_"))

    @property
    def reserve_cubes(self) -> list[str]:
        """Names of cube objects currently registered in this scene config."""
        return sorted(k for k in self.__dict__ if k.startswith("reserve_cube_"))

    @property
    def colors_amount(self) -> int:
        """Number of colors used for cubes in this scene config."""
        return 4

    @property
    def cubes_size(self) -> tuple[float, float, float]:
        """Size of cubes used in this scene config."""
        return (0.05, 0.05, 0.05)  # (x, y, z) in meters

    @property
    def separation_distance(self) -> float:
        """Minimum separation distance between cubes in this scene config."""
        return 0.05  # in meters

    @property
    def reserve_cubes_amount_per_color(self) -> int:
        """Number of reserve cubes used per color in this scene config."""
        return 12

    @property
    def _reserve_cubes_amount_per_line(self) -> int:
        """Number of reserve cubes used per line in this scene config."""
        return 3

    def change_reserve_cubes_colors(self, colors: list[Colors] = DEFAULT_COLORS) -> None:
        if len(colors) < self.colors_amount:
            raise ValueError(f"Expected at least {self.colors_amount} colors, got {len(colors)}")

        for cube_name in self.reserve_cubes:
            parts = cube_name.split("_")
            if len(parts) < 4:
                continue

            color_idx = int(parts[2])
            cube_cfg = getattr(self, cube_name)

            if isinstance(cube_cfg.spawn, UsdFileCfg):
                cube_cfg.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{colors[color_idx].value}_block.usd"

    def initialize_reserve_cubes(self, colors: list[Colors] = DEFAULT_COLORS) -> None:
        x_offsets = [0.2, 0.5]
        y_offsets = [-0.1, 0.1]
        # init reserve cubes
        for color_idx in range(self.colors_amount):
            for i in range(self.reserve_cubes_amount_per_color):
                offset_x = x_offsets[int(color_idx / 2)]
                offset_y = y_offsets[color_idx % 2]
                x = offset_x + (i % self._reserve_cubes_amount_per_line) * (
                    self.cubes_size[0] + self.separation_distance
                )
                y = offset_y
                cube_name = f"reserve_cube_{colors[color_idx].value}_{i+1}"
                setattr(
                    self,
                    cube_name,
                    RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/" + cube_name,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(x, y, 0.02 + ((i // self._reserve_cubes_amount_per_line) * (self.cubes_size[2])))
                        ),
                        spawn=UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{colors[color_idx].value}_block.usd",
                            semantic_tags=[("class", cube_name)],
                            scale=(1.0, 1.0, 1.0),
                        ),
                    ),
                )

    def get_reserve_cube_poses(self) -> list[tuple[float, float, float, float, float, float, float]]:
        """Get the poses of all reserve cubes in this scene config."""
        poses = []
        for cube_name in self.reserve_cubes:
            cube_cfg: RigidObjectCfg = getattr(self, cube_name)
            if isinstance(cube_cfg.spawn, UsdFileCfg):
                poses.append(cube_cfg.init_state.pos + cube_cfg.init_state.rot)
        return poses

    def get_reserve_cube_colors(self) -> list[Colors]:
        """Get the colors of all reserve cubes in this scene config."""
        colors = []
        for cube_name in self.reserve_cubes:
            parts = cube_name.split("_")
            if len(parts) < 4:
                continue
            colors.append(Colors(parts[2]))
        return colors

    def get_reserve_cube_stacks(self) -> dict[Colors, torch.Tensor]:
        """Get the poses of all reserve cubes in this scene config, grouped by color."""
        stacks = {}
        cube_poses = self.get_reserve_cube_poses()
        names = self.reserve_cubes
        cube_colors = self.get_reserve_cube_colors()

        name_color_pose = list(zip(names, cube_colors, cube_poses))
        sorted_by_idx = sorted(name_color_pose, key=lambda x: int(x[0].split("_")[-1]))
        for color in set(cube_colors):
            stacks[color] = torch.tensor([pose for _, c, pose in sorted_by_idx if c == color])

        return stacks

    def get_reserve_cube_name(self, color: Colors, index: int) -> str:
        """Get the name of a reserve cube given its color and index."""
        return f"reserve_cube_{color.value}_{index+1}"

    def __post_init__(self):
        # return
        self.initialize_reserve_cubes()
