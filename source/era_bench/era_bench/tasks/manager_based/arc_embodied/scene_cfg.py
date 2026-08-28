from dataclasses import MISSING
from enum import Enum
import math

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG  # isort: skip
from arc_utils import ARC_COLOR_TO_COLOR, COLOR_RGB, COLOR_TO_ARC_COLOR, Colors
from pxr import Semantics

FRANKA_GRASP_CENTER_OFFSET = (0.0, 0.0, 0.1034)


class GridArea(Enum):
    """The two fixed ARC grid areas in every environment."""

    INPUT = "input"
    OUTPUT = "output"


def _linear_rgb_to_srgb_rgba8(linear_rgb: tuple[float, float, float]) -> tuple[int, int, int, int]:
    """Convert the canonical linear ARC material color to an 8-bit segmentation color."""

    def to_srgb8(channel: float) -> int:
        srgb = 12.92 * channel if channel <= 0.0031308 else 1.055 * math.pow(channel, 1.0 / 2.4) - 0.055
        return round(255 * srgb)

    return (*map(to_srgb8, linear_rgb), 255)


SEMANTIC_SEGMENTATION_MAPPING = {
    **{
        f"class:cube_{color.value}": _linear_rgb_to_srgb_rgba8(COLOR_RGB[color])
        for color in ARC_COLOR_TO_COLOR.values()
    },
    "class:cube_unassigned": (80, 80, 80, 255),
    "class:table": (255, 237, 218, 255),
    "class:ground": (100, 100, 100, 255),
    "class:robot": (204, 110, 248, 255),
    "class:grid_background": (0, 0, 0, 255),
    "class:reserve_grid_background": (58, 31, 92, 255),
    "class:grid_line": (255, 255, 255, 255),
    "class:UNLABELLED": (150, 150, 150, 255),
    "class:BACKGROUND": (0, 0, 0, 255),
}
"""Stable RGBA legend used by every colorized semantic-segmentation camera."""


@configclass
class EraBenchSceneCfg(InteractiveSceneCfg):

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=112,
        width=112,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        depth_clipping_behavior="max",
        semantic_filter=["class"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping=SEMANTIC_SEGMENTATION_MAPPING,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=27.294820675, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    )

    table_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=252,
        width=336,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        depth_clipping_behavior="max",
        semantic_filter=["class"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping=SEMANTIC_SEGMENTATION_MAPPING,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0, focus_distance=2.3, horizontal_aperture=18.43, clipping_range=(0.1, 5.0)
        ),
        # offset=TiledCameraCfg.OffsetCfg(
        #     pos=(1.8, 0.0, 0.7), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
        # ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.43, 0.0, 1.6), rot=(0.23021145, -0.66858260, -0.66858260, 0.23021145), convention="ros"
        ),
    )

    origin_top_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/origin_top_cam",
        update_period=0.0,
        height=280,
        width=280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.294922600, focus_distance=2.4, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
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
        """Neutral reserve-cube names in physical stack order."""
        return [
            self.get_reserve_cube_name(stack_index, cube_index)
            for stack_index in range(self.reserve_stacks_amount)
            for cube_index in range(self.reserve_cubes_amount_per_stack)
        ]

    @property
    def input_grid_cubes(self) -> list[str]:
        """Names of the pooled, static cubes used to display an ARC input grid."""
        return sorted(k for k in self.__dict__ if k.startswith("input_grid_cube_"))

    @property
    def reserve_stacks_amount(self) -> int:
        """Number of physical reserve stacks available in every environment."""
        return 4

    @property
    def cubes_size(self) -> tuple[float, float, float]:
        """Size of cubes used in this scene config."""
        return (0.04, 0.04, 0.04)  # (x, y, z) in meters

    @property
    def separation_distance(self) -> float:
        """Minimum separation distance between cubes in this scene config."""
        return 0.02  # in meters

    @property
    def reserve_cubes_amount_per_stack(self) -> int:
        """Number of cubes in each physical reserve stack."""
        return 12

    @property
    def reserve_cubes_amount_per_line(self) -> int:
        """Number of reserve cubes used per line in this scene config."""
        return 4

    @property
    def reserve_stack_x_local(self) -> float:
        """Common local x coordinate of all reserve-stack starting points."""
        return 0.30

    @property
    def reserve_stack_y_start_local(self) -> float:
        """Local y coordinate of the first reserve-stack starting point."""
        return -0.1

    @property
    def reserve_stack_y_spacing(self) -> float:
        """Local y spacing between reserve stacks."""
        return 0.2

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

    @property
    def grid_cell_size(self) -> float:
        """Center-to-center pitch of one displayed ARC grid cell in meters."""
        return self.cubes_size[0] + self.separation_distance

    @property
    def grid_line_width(self) -> float:
        """Width of the visual grid boundaries in meters."""
        return self.separation_distance / 4.0

    @property
    def grid_visual_height(self) -> float:
        """Thickness of the visual-only grid background and boundary prims in meters."""
        return 0.0005

    @property
    def max_grid_size(self) -> int:
        """Largest supported ARC grid dimension for the visual line pools."""
        return 8

    def get_grid_center_local(self, grid_area: GridArea) -> tuple[float, float, float]:
        """Return the configured anchor for an input or output ARC grid area."""
        if grid_area is GridArea.INPUT:
            return self.input_grid_center_local
        if grid_area is GridArea.OUTPUT:
            return self.output_grid_center_local
        raise ValueError(f"Unknown grid area: {grid_area!r}")

    def get_grid_background_name(self, grid_area: GridArea) -> str:
        """Return the stable visual-background name for one grid area."""
        self.get_grid_center_local(grid_area)
        return f"{grid_area.value}_grid_background"

    def get_grid_row_line_name(self, grid_area: GridArea, line_index: int) -> str:
        """Return one row-boundary visual line name."""
        self.get_grid_center_local(grid_area)
        if not 0 <= line_index <= self.max_grid_size:
            raise ValueError(f"Invalid grid row line index: {line_index}")
        return f"{grid_area.value}_grid_row_line_{line_index}"

    def get_grid_column_line_name(self, grid_area: GridArea, line_index: int) -> str:
        """Return one column-boundary visual line name."""
        self.get_grid_center_local(grid_area)
        if not 0 <= line_index <= self.max_grid_size:
            raise ValueError(f"Invalid grid column line index: {line_index}")
        return f"{grid_area.value}_grid_column_line_{line_index}"

    @property
    def reserve_grid_background_name(self) -> str:
        """Stable visual-only background name for the reserve-cube area."""
        return "reserve_grid_background"

    def get_reserve_grid_background_center_local(self) -> tuple[float, float, float]:
        """Return the center of the complete reserve-cube footprint."""
        pitch = self.grid_cell_size
        stack_starts = [
            self.get_reserve_stack_start_local(stack_index) for stack_index in range(self.reserve_stacks_amount)
        ]
        first_x = min(x for x, _ in stack_starts)
        last_x = max(x for x, _ in stack_starts) + (self.reserve_cubes_amount_per_line - 1) * pitch
        first_y = min(y for _, y in stack_starts)
        last_y = max(y for _, y in stack_starts)
        return ((first_x + last_x) / 2, (first_y + last_y) / 2, self.grid_visual_height / 2)

    def get_reserve_grid_background_size(self) -> tuple[float, float, float]:
        """Return the visual reserve-area footprint including a half-cell border."""
        pitch = self.grid_cell_size
        stack_starts = [
            self.get_reserve_stack_start_local(stack_index) for stack_index in range(self.reserve_stacks_amount)
        ]
        first_x = min(x for x, _ in stack_starts)
        last_x = max(x for x, _ in stack_starts) + (self.reserve_cubes_amount_per_line - 1) * pitch
        first_y = min(y for _, y in stack_starts)
        last_y = max(y for _, y in stack_starts)
        return (last_x - first_x + pitch, last_y - first_y + pitch, self.grid_visual_height)

    def get_reserve_stack_start_local(self, stack_index: int) -> tuple[float, float]:
        """Return the local starting position of a physical reserve stack."""
        if not 0 <= stack_index < self.reserve_stacks_amount:
            raise ValueError(f"Invalid reserve stack index: {stack_index}")
        return (
            self.reserve_stack_x_local
            + (
                self.reserve_cubes_amount_per_line
                * (self.cubes_size[0] + self.separation_distance)
                * int(stack_index / 2 % 2)
            ),
            self.reserve_stack_y_start_local + (stack_index % 2) * self.reserve_stack_y_spacing,
        )

    def get_input_grid_cubes_amount(self, arc_color: int) -> int:
        """Return the number of pooled input-grid cubes available for an ARC colour ID."""
        if arc_color not in ARC_COLOR_TO_COLOR:
            raise ValueError(f"Unsupported ARC colour ID: {arc_color}")
        if ARC_COLOR_TO_COLOR[arc_color] is Colors.BLUE:
            return self.input_grid_blue_cubes_amount
        return self.input_grid_cubes_amount_per_color

    def _cube_visual_material(self, color: Colors) -> sim_utils.PreviewSurfaceCfg:
        return sim_utils.PreviewSurfaceCfg(diffuse_color=COLOR_RGB[color])

    def get_cube_semantic_label(self, arc_color: int) -> str:
        """Return the color-aware semantic class used for an ARC cube."""
        if arc_color not in ARC_COLOR_TO_COLOR:
            raise ValueError(f"Unsupported ARC colour ID: {arc_color}")
        return f"cube_{ARC_COLOR_TO_COLOR[arc_color].value}"

    def initialize_reserve_cubes(self) -> None:

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
        # Reserve assets have stable, color-agnostic identities. Reset events
        # assign their per-environment colors and poses.
        for stack_index in range(self.reserve_stacks_amount):
            for cube_index in range(self.reserve_cubes_amount_per_stack):
                cube_name = self.get_reserve_cube_name(stack_index, cube_index)
                setattr(
                    self,
                    cube_name,
                    RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/" + cube_name,
                        spawn=sim_utils.CuboidCfg(
                            size=self.cubes_size,
                            semantic_tags=[("class", "cube_unassigned")],
                            rigid_props=reserve_cube_rigid_props,
                            collision_props=reserve_cube_collision_props,
                            mass_props=reserve_cube_mass_props,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
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
                            semantic_tags=[("class", self.get_cube_semantic_label(arc_color))],
                            rigid_props=input_cube_rigid_props,
                            # No collision_props: input cubes are visual references only.
                            visual_material=self._cube_visual_material(color),
                        ),
                    ),
                )

    def initialize_grid_visuals(self) -> None:
        """Create visual-only black grid backgrounds and white boundary-line pools."""
        cell_size = self.grid_cell_size
        visual_height = self.grid_visual_height
        for grid_area in GridArea:
            center_x, center_y, _ = self.get_grid_center_local(grid_area)
            # Preserve the established cube placement, whose cell-center average
            # lies half a cube width below the configured grid anchor.
            visual_center = (center_x - self.cubes_size[0] / 2, center_y - self.cubes_size[1] / 2)
            setattr(
                self,
                self.get_grid_background_name(grid_area),
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/" + self.get_grid_background_name(grid_area),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(*visual_center, visual_height / 2)),
                    spawn=sim_utils.CuboidCfg(
                        size=(cell_size, cell_size, visual_height),
                        semantic_tags=[("class", "grid_background")],
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
                    ),
                ),
            )
            for line_index in range(self.max_grid_size + 1):
                setattr(
                    self,
                    self.get_grid_row_line_name(grid_area, line_index),
                    AssetBaseCfg(
                        prim_path="{ENV_REGEX_NS}/" + self.get_grid_row_line_name(grid_area, line_index),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(*visual_center, visual_height * 1.5)),
                        spawn=sim_utils.CuboidCfg(
                            size=(self.grid_line_width, cell_size, visual_height),
                            semantic_tags=[("class", "grid_line")],
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                        ),
                    ),
                )
                setattr(
                    self,
                    self.get_grid_column_line_name(grid_area, line_index),
                    AssetBaseCfg(
                        prim_path="{ENV_REGEX_NS}/" + self.get_grid_column_line_name(grid_area, line_index),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(*visual_center, visual_height * 1.5)),
                        spawn=sim_utils.CuboidCfg(
                            size=(cell_size, self.grid_line_width, visual_height),
                            semantic_tags=[("class", "grid_line")],
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                        ),
                    ),
                )

    def initialize_reserve_grid_visual(self) -> None:
        """Create the visual-only violet background beneath the reserve-cube area."""
        setattr(
            self,
            self.reserve_grid_background_name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + self.reserve_grid_background_name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=self.get_reserve_grid_background_center_local()),
                spawn=sim_utils.CuboidCfg(
                    size=self.get_reserve_grid_background_size(),
                    semantic_tags=[("class", "reserve_grid_background")],
                    # Linear RGB equivalent of the sRGB deep violet #3A1F5C.
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.042, 0.014, 0.107)),
                ),
            ),
        )

    def get_reserve_cube_name(self, stack_index: int, cube_index: int) -> str:
        """Return a stable neutral reserve-cube name from physical indices."""
        if not 0 <= stack_index < self.reserve_stacks_amount:
            raise ValueError(f"Invalid reserve stack index: {stack_index}")
        if not 0 <= cube_index < self.reserve_cubes_amount_per_stack:
            raise ValueError(f"Invalid reserve cube index: {cube_index}")
        slot_index = stack_index * self.reserve_cubes_amount_per_stack + cube_index
        return f"reserve_cube_{slot_index:02d}"

    def get_reserve_cube_shader(self, env_id: int, stack_index: int, cube_index: int):
        """Return the shader prim for one environment-local reserve cube."""
        cube_name = self.get_reserve_cube_name(stack_index, cube_index)
        shader_path = f"/World/envs/env_{env_id}/{cube_name}/geometry/material/Shader"
        shader = get_current_stage().GetPrimAtPath(shader_path)
        if not shader.IsValid():
            raise RuntimeError(f"Reserve cube shader does not exist: {shader_path}")
        return shader

    def get_reserve_cube_semantics(self, env_id: int, stack_index: int, cube_index: int):
        """Return the class semantics API for one environment-local reserve cube."""
        cube_name = self.get_reserve_cube_name(stack_index, cube_index)
        cube_path = f"/World/envs/env_{env_id}/{cube_name}"
        cube_prim = get_current_stage().GetPrimAtPath(cube_path)
        if not cube_prim.IsValid():
            raise RuntimeError(f"Reserve cube prim does not exist: {cube_path}")
        semantics = Semantics.SemanticsAPI(cube_prim, "class_cube_unassigned")
        if not semantics.GetSemanticDataAttr().IsValid():
            raise RuntimeError(f"Reserve cube class semantics do not exist: {cube_path}")
        return semantics

    def get_reserve_cube_arc_color(self, env_id, stack_index: int, cube_index: int) -> int:
        semantic_label = self.get_reserve_cube_semantics(env_id, stack_index, cube_index).GetSemanticDataAttr().Get()
        return COLOR_TO_ARC_COLOR[Colors(semantic_label.removeprefix("cube_"))]

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

    def __post_init__(self):
        self.initialize_reserve_cubes()
        self.initialize_input_grid_cubes()
        self.initialize_grid_visuals()
        self.initialize_reserve_grid_visual()
