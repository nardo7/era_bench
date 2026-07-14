from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG  # isort: skip
from arc_utils import ARC_COLOR_TO_COLOR, COLOR_RGB, DEFAULT_COLORS, Colors
import torch

FRANKA_GRASP_CENTER_OFFSET = (0.0, 0.0, 0.1034)


@configclass
class EraBenchSceneCfg(InteractiveSceneCfg):

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    )

    table_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=256,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.8, 0.0, 0.7), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
        ),
        # offset=CameraCfg.OffsetCfg(pos=(1.8, 0.0, 1.7), rot=(0.24184, -0.66446, -0.66446, 0.24184), convention="ros"),
    )

    origin_top_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/origin_top_cam",
        update_period=0.0,
        height=384,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.2, 0.0, 2.4), rot=(0.0, 0.70711, 0.70711, 0.0), convention="ros"),
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
    def input_grid_cubes(self) -> list[str]:
        """Names of the pooled, static cubes used to display an ARC input grid."""
        return sorted(k for k in self.__dict__ if k.startswith("input_grid_cube_"))

    @property
    def colors_amount(self) -> int:
        """Number of colors used for cubes in this scene config."""
        return 2

    @property
    def cubes_size(self) -> tuple[float, float, float]:
        """Size of cubes used in this scene config."""
        return (0.04, 0.04, 0.04)  # (x, y, z) in meters

    @property
    def separation_distance(self) -> float:
        """Minimum separation distance between cubes in this scene config."""
        return 0.02  # in meters

    @property
    def reserve_cubes_amount_per_color(self) -> int:
        """Number of reserve cubes used per color in this scene config."""
        return 9

    @property
    def reserve_cubes_amount_per_line(self) -> int:
        """Number of reserve cubes used per line in this scene config."""
        return 3

    @property
    def input_grid_cubes_amount_per_color(self) -> int:
        """Default number of pooled reference cubes for each ARC colour."""
        return 32

    @property
    def input_grid_blue_cubes_amount(self) -> int:
        """Number of pooled reference cubes reserved for ARC blue (colour ID 1)."""
        return 45

    @property
    def input_grid_center_local(self) -> tuple[float, float, float]:
        """Local position of the center of the input grid in this scene config."""
        return (0.0, -0.5, 0.00)  # (x, y, z) in meters

    @property
    def output_grid_center_local(self) -> tuple[float, float, float]:
        """Local position of the center of the output grid in this scene config."""
        return (0.0, 0.5, 0.00)  # (x, y, z) in meters

    def get_input_grid_cubes_amount(self, arc_color: int) -> int:
        """Return the number of pooled input-grid cubes available for an ARC colour ID."""
        if arc_color not in ARC_COLOR_TO_COLOR:
            raise ValueError(f"Unsupported ARC colour ID: {arc_color}")
        if ARC_COLOR_TO_COLOR[arc_color] is Colors.BLUE:
            return self.input_grid_blue_cubes_amount
        return self.input_grid_cubes_amount_per_color

    def _cube_visual_material(self, color: Colors) -> sim_utils.PreviewSurfaceCfg:
        return sim_utils.PreviewSurfaceCfg(diffuse_color=COLOR_RGB[color])

    def change_reserve_cubes_colors(self, colors: list[Colors] = DEFAULT_COLORS) -> None:
        if len(colors) < self.colors_amount:
            raise ValueError(f"Expected at least {self.colors_amount} colors, got {len(colors)}")

        for cube_name in self.reserve_cubes:
            parts = cube_name.split("_")
            if len(parts) < 4:
                continue

            color_idx = int(parts[2])
            cube_cfg = getattr(self, cube_name)
            cube_cfg.spawn.visual_material = self._cube_visual_material(colors[color_idx])

    def initialize_reserve_cubes(self, colors: list[Colors] = DEFAULT_COLORS) -> None:

        reserve_cube_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=1.0,
            linear_damping=1.0,
            angular_damping=1.0,
            disable_gravity=False,
        )
        reserve_cube_collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0025,
            rest_offset=0.0,
        )
        reserve_cube_mass_props = sim_utils.MassPropertiesCfg(
            mass=0.08,
        )
        # init reserve cubes
        for color_idx in range(self.colors_amount):
            for i in range(self.reserve_cubes_amount_per_color):
                cube_name = f"reserve_cube_{colors[color_idx].value}_{i+1}"
                setattr(
                    self,
                    cube_name,
                    RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/" + cube_name,
                        spawn=sim_utils.CuboidCfg(
                            size=self.cubes_size,
                            semantic_tags=[("class", cube_name)],
                            rigid_props=reserve_cube_rigid_props,
                            collision_props=reserve_cube_collision_props,
                            mass_props=reserve_cube_mass_props,
                            visual_material=self._cube_visual_material(colors[color_idx]),
                        ),
                    ),
                )

    def initialize_input_grid_cubes(self) -> None:
        """Create a static, reset-reusable pool for all possible ARC input pixels."""
        input_cube_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )

        for arc_color, color in ARC_COLOR_TO_COLOR.items():
            for i in range(self.get_input_grid_cubes_amount(arc_color)):
                cube_name = self.get_input_grid_cube_name(arc_color, i)
                setattr(
                    self,
                    cube_name,
                    RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/" + cube_name,
                        spawn=sim_utils.CuboidCfg(
                            size=self.cubes_size,
                            semantic_tags=[("class", cube_name)],
                            rigid_props=input_cube_rigid_props,
                            # No collision_props: input cubes are visual references only.
                            visual_material=self._cube_visual_material(color),
                        ),
                    ),
                )

    def get_reserve_cube_poses(self) -> list[tuple[float, float, float, float, float, float, float]]:
        """Get the poses of all reserve cubes in this scene config."""
        poses = []
        for cube_name in self.reserve_cubes:
            cube_cfg: RigidObjectCfg = getattr(self, cube_name)
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
        ordered_colors = [color for color in DEFAULT_COLORS if color in cube_colors]
        for color in ordered_colors:
            stacks[color] = torch.tensor([pose for _, c, pose in sorted_by_idx if c == color])

        return stacks

    def get_reserve_cube_name(self, color: Colors, index: int) -> str:
        """Get the name of a reserve cube given its color and index."""
        return f"reserve_cube_{color.value}_{index+1}"

    def get_input_grid_cube_name(self, arc_color: int, index: int) -> str:
        """Return the pooled input-cube name for an ARC colour ID and zero-based slot."""
        if arc_color not in ARC_COLOR_TO_COLOR:
            raise ValueError(f"Unsupported ARC colour ID: {arc_color}")
        pool_size = self.get_input_grid_cubes_amount(arc_color)
        if not 0 <= index < pool_size:
            raise ValueError(
                f"Input-grid cube index {index} exceeds the pool size " f"({pool_size}) for ARC colour {arc_color}."
            )
        return f"input_grid_cube_{ARC_COLOR_TO_COLOR[arc_color].value}_{index+1}"

    def get_reserve_cube_color(self, cube_name: str) -> Colors:
        """Get the color of a reserve cube given its name."""
        parts = cube_name.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid reserve cube name: {cube_name}")
        return Colors(parts[2])

    def __post_init__(self):
        self.initialize_reserve_cubes()
        self.initialize_input_grid_cubes()
