from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
import random
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


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
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(2.2, 0.0, 1.2), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"),
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
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.8), rot=(0.0, 0.70711, 0.70711, 0.0), convention="ros"),
    )

    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING  # pyright: ignore[reportAssignmentType]

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0, -0.05),
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

    def __post_init__(self):
        # Define a thin physical barrier
        line_props = sim_utils.CuboidCfg(
            size=(0.5, 0.05, 0.05),  # Long (X), narrow (Y), very short height (Z)
            # Enable physical collisions and dynamics
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # Set True to keep it locked in place permanently
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            # Visual color (e.g., bright yellow for a boundary line)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
        )

        self.line = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + self.line_name,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.01), rot=(1, 0, 0, 0)),
            spawn=line_props,
        )

        objects_amount = random.randint(2, 3)

        for i in range(objects_amount):
            cube_name = f"cube_{i+1}"
            setattr(
                self,
                cube_name,
                RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + cube_name,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.02)),
                    spawn=UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                        semantic_tags=[("class", cube_name)],
                        scale=(1.0, 1.0, 1.0),
                    ),
                ),
            )
