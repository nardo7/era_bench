"""Extended cuRobo planner with plan_goalset support."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, cast

import torch
import torch.nn.functional as torch_f
from curobo.types.math import Pose
from curobo.types.state import JointState
import isaaclab.utils.math as math_utils
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner


class ExtendedCuroboPlanner(CuroboPlanner):
    """CuroboPlanner subclass that adds plan_goalset for batch goal evaluation.

    Attempts to use cuRobo's native plan_goalset / plan_batch_goalset APIs when
    available. Falls back to sequential per-goal planning for safety.
    """

    _MAX_TASK_TRAJECTORY_DISTANCE = torch.linalg.vector_norm(
        torch.tensor([0.8, -0.1, 0.25]) - torch.tensor([-0.33, 0.73, 0.025])
    )
    _PHASE_LOG_FILENAME = "curobo_phase_debug.jsonl"

    def __init__(
        self,
        *args,
        grasp_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_native_goalset: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._grasp_center_offset = grasp_center_offset
        self._expected_attached_object: str | None = None
        self._use_native_goalset = use_native_goalset
        self._phase_log_path = self._default_phase_log_path()

    def _default_phase_log_path(self) -> Path:
        """Resolve phase debug logs to the project-level logs directory."""
        module_path = Path(__file__).resolve()
        if len(module_path.parents) >= 4:
            return module_path.parents[3] / "logs" / self._PHASE_LOG_FILENAME
        return Path.cwd() / "logs" / self._PHASE_LOG_FILENAME

    def set_expected_attached_object(self, object_name: str | None) -> None:
        self._expected_attached_object = object_name

    def grasp_center_pose_to_panda_hand_pose(self, target_pose: torch.Tensor) -> torch.Tensor:
        """Convert a grasp-center goal pose into the corresponding panda_hand goal pose."""
        target_pos = target_pose[..., :3]
        target_quat = target_pose[..., 3:7]
        offset_pos = torch.tensor(self._grasp_center_offset, device=target_pose.device, dtype=target_pose.dtype)
        offset_pos = offset_pos.expand_as(target_pos)
        offset_quat = torch.zeros_like(target_quat)
        offset_quat[..., 0] = 1.0

        offset_inv_pos, offset_inv_quat = math_utils.subtract_frame_transforms(
            offset_pos,
            offset_quat,
        )
        hand_pos, hand_quat = math_utils.combine_frame_transforms(
            target_pos,
            target_quat,
            offset_inv_pos,
            offset_inv_quat,
        )
        return torch.cat((hand_pos, hand_quat), dim=-1)

    def attach_object(self, object_name: str, gripper_pose: torch.Tensor, env_id: int = 0) -> None:
        """Attach an object to the robot's end-effector for planning.

        Args:
            object_name: Name of the object to attach.
            gripper_pose: Pose of the gripper in world frame.
        """
        self.set_expected_attached_object(object_name)
        self.update_world()
        self._sync_expected_attachment(env_id)

    def detach_objects(self) -> None:
        self.set_expected_attached_object(None)
        self._detach_objects()

    def panda_hand_pose_to_grasp_center_pose(self, target_pose: torch.Tensor) -> torch.Tensor:
        """Convert a panda_hand pose into the corresponding grasp-center pose."""
        target_pos = target_pose[..., :3, 3]
        target_rot = target_pose[..., :3, :3]
        target_quat = math_utils.quat_from_matrix(target_rot)
        offset_pos = torch.tensor(self._grasp_center_offset, device=target_pose.device, dtype=target_pose.dtype)
        offset_pos = offset_pos.expand_as(target_pos)
        offset_quat = torch.zeros_like(target_quat)
        offset_quat[..., 0] = 1.0
        grasp_pos, grasp_quat = math_utils.combine_frame_transforms(
            target_pos,
            target_quat,
            offset_pos,
            offset_quat,
        )
        grasp_rot = math_utils.matrix_from_quat(grasp_quat)
        return math_utils.make_pose(grasp_pos, grasp_rot)

    def plan_grasp_center_goal_with_fixed_yaw_candidates(
        self,
        grasp_center_goal: torch.Tensor,
        num_candidates: int = 8,
        contact_window_fraction: float | None = 0.15,
    ) -> bool:
        """Plan to a grasp-center goal by converting it to the panda_hand frame expected by cuRobo."""
        planner_pose = self.grasp_center_pose_to_panda_hand_pose(grasp_center_goal.unsqueeze(0))[0]
        planner_base_quat = planner_pose[3:7]
        return self.plan_with_fixed_yaw_candidates(
            planner_pose[:3],
            planner_base_quat,
            num_candidates=num_candidates,
            contact_window_fraction=contact_window_fraction,
        )

    def plan_with_fixed_yaw_candidates(
        self,
        target_position: torch.Tensor,
        base_quat: torch.Tensor,
        num_candidates: int = 8,
        contact_window_fraction: float | None = 0.15,
    ) -> bool:
        """Try a fixed set of wrist-yaw variants and keep the first feasible plan."""
        candidate_poses = self._generate_yaw_candidates(
            target_position=target_position,
            base_quat=base_quat,
            num_candidates=num_candidates,
        )
        candidate_poses = torch.stack(candidate_poses, dim=0)
        return self.plan_goalset(
            candidate_poses[:, :3],
            candidate_poses[:, 3:7],
            contact_window_fraction=contact_window_fraction,
        )

    def plan_goalset(
        self,
        positions: Iterable[torch.Tensor],
        quaternions: Iterable[torch.Tensor],
        contact_window_fraction: float | None = None,
    ) -> bool:
        """Plan trajectory to the best reachable goal from a set of candidates.

        Uses sequential per-goal planning by default because the native cuRobo
        goalset API changes the internal solve type to GOALSET, which can fail on
        this runtime when cached CUDA graph state cannot be reset. The native API
        can still be enabled explicitly for environments where that path is stable.

        Args:
            positions: Iterable of 3-element position tensors (EE goal positions).
            quaternions: Iterable of 4-element quaternion tensors (w,x,y,z format).
            contact_window_fraction: When given (0 < f < 0.5), each fallback candidate
                is planned in three phases:
                  1. start → interpolated_entry  (fraction of the way, contact=True)
                  2. entry → interpolated_exit   (collision-checked middle, contact=False)
                  3. exit  → goal                (contact=True)
                This mirrors Isaac Lab's multi-phase logic but uses Cartesian fractions
                instead of fixed metric distances.

        Returns:
            True if a trajectory was found, False otherwise.
        """
        positions = list(positions)
        quaternions = list(quaternions)
        poses = [self._make_pose(position=pos, quaternion=quat) for pos, quat in zip(positions, quaternions)]
        if not poses:
            self.logger.warning("plan_goalset: empty goal set")
            return False

        goal_positions = torch.stack(positions, dim=0).unsqueeze(0)
        goal_quaternions = torch.stack(quaternions, dim=0).unsqueeze(0)
        goal_pose = Pose(position=goal_positions, quaternion=goal_quaternions)

        self.reset_plan()
        self.update_world()
        self._sync_expected_attachment(env_id=0)
        start_state = self._get_current_joint_state_for_curobo()

        try:
            if self._use_native_goalset:
                mg = self.motion_gen
                for api_name in ("plan_goalset", "plan_batch_goalset"):
                    fn = getattr(mg, api_name, None)
                    if fn is None:
                        continue
                    try:
                        result = fn(start_state, goal_pose, self.plan_config)
                        if result is not None and self._result_succeeded(result):
                            plan = self._extract_plan(result)
                            if plan is not None:
                                self._current_plan = self._finalise_plan(plan)
                                self._plan_index = 0
                                self.logger.info(
                                    f"plan_goalset native succeeded: {len(self._current_plan.position)} waypoints"
                                )
                                return True
                    except Exception as exc:
                        self.logger.debug(f"{api_name} failed: {exc}")

                self.logger.info("plan_goalset: native failed, falling back to sequential planning")
            else:
                self.logger.info("plan_goalset: native goalset disabled, using sequential planning")

            for pose in poses:
                try:
                    success = (
                        self._plan_to_contact_with_window(
                            start_state=start_state,
                            goal_pose=pose,
                            contact_window_fraction=contact_window_fraction,
                        )
                        if contact_window_fraction is not None
                        else self._plan_to_contact(
                            start_state=start_state,
                            goal_pose=pose,
                            retreat_distance=self.config.retreat_distance,
                            approach_distance=self.config.approach_distance,
                            contact=False,
                        )
                    )
                    if success:
                        return True
                except Exception:
                    continue

            return False
        finally:
            self._detach_objects()

    def _sync_expected_attachment(self, env_id: int) -> None:
        expected_attached_object = self._expected_attached_object
        gripper_closed = expected_attached_object is not None
        self._set_gripper_state(gripper_closed)
        current_attached = self.get_attached_objects()

        if expected_attached_object and expected_attached_object not in current_attached:
            object_mappings = self._get_object_mappings()
            if expected_attached_object in object_mappings:
                gripper_pos = self.robot.data.joint_pos[env_id, -2:]
                is_grasped = self._check_object_grasped(gripper_pos, expected_attached_object)
                if is_grasped:
                    self._attach_object(expected_attached_object, object_mappings[expected_attached_object], env_id)

        if expected_attached_object is None and current_attached:
            self._detach_objects()

    def _generate_yaw_candidates(
        self,
        target_position: torch.Tensor,
        base_quat: torch.Tensor,
        num_candidates: int,
    ) -> list[torch.Tensor]:
        """Generate fixed wrist-yaw variants for a target position."""
        yaw_angles = torch.linspace(
            0.0,
            2.0 * torch.pi,
            steps=num_candidates + 1,
            device=target_position.device,
            dtype=target_position.dtype,
        )[:-1]
        zeros = torch.zeros_like(yaw_angles)
        yaw_quats = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_angles)
        candidate_quats = math_utils.quat_mul(yaw_quats, base_quat.unsqueeze(0).expand(num_candidates, -1))
        return [torch.cat((target_position, quat), dim=0) for quat in candidate_quats]

    def _plan_to_contact_with_window(
        self,
        start_state: JointState,
        goal_pose: Pose,
        contact_window_fraction: float,
    ) -> bool:
        """Multi-phase planning using Cartesian contact windows.

        Phase 1 (contact=True):  start → pose at fraction  of the way to goal.
        Phase 2 (contact=False): entry → pose at 1-fraction of the way to goal.
        Phase 3 (contact=True):  exit  → goal.

        contact=True disables hand/finger collision spheres (same as Isaac Lab's
        multi-phase planner), allowing the solver to move through near-contact
        geometry at the start and end of the trajectory.
        """
        if not 0.0 < contact_window_fraction < 0.5:
            raise ValueError(f"contact_window_fraction must be in (0.0, 0.5), got {contact_window_fraction}")

        start_pose = self.get_ee_pose(start_state)
        entry_pose = self._interpolate_pose(start_pose, goal_pose, contact_window_fraction)
        exit_pose = self._interpolate_pose(start_pose, goal_pose, 1.0 - contact_window_fraction)
        entry_steps, middle_steps, exit_steps = self._phase_step_budget(
            start_pose=start_pose,
            goal_pose=goal_pose,
            contact_window_fraction=contact_window_fraction,
        )
        self._log_phase_calculation(
            phase="budget",
            success=True,
            contact=False,
            start_position=self._pose_position_to_list(start_pose),
            goal_position=self._pose_position_to_list(goal_pose),
            distance=self._cartesian_distance(start_pose, goal_pose),
            contact_window_fraction=contact_window_fraction,
            allocated_steps={"entry": entry_steps, "middle": middle_steps, "exit": exit_steps},
        )

        # Phase 1: contact-allowed start window
        phase1_success = self._plan_to_contact_pose(start_state=start_state, goal_pose=entry_pose, contact=True)
        self._log_phase_calculation(
            phase="entry",
            success=phase1_success,
            contact=True,
            start_position=self._pose_position_to_list(start_pose),
            goal_position=self._pose_position_to_list(entry_pose),
            allocated_steps=entry_steps,
        )
        if not phase1_success:
            self.logger.debug("_plan_to_contact_with_window: phase 1 (entry) failed")
            return False
        assert self._current_plan is not None
        phase1 = self._resample_plan(self._current_plan, entry_steps)

        # Phase 2: collision-checked middle
        mid_start = self._joint_state_from_plan_end(phase1)
        mid_start_pose = self.get_ee_pose(mid_start)
        phase2_success = self._plan_to_contact_pose(start_state=mid_start, goal_pose=exit_pose, contact=False)
        self._log_phase_calculation(
            phase="middle",
            success=phase2_success,
            contact=False,
            start_position=self._pose_position_to_list(mid_start_pose),
            goal_position=self._pose_position_to_list(exit_pose),
            allocated_steps=middle_steps,
        )
        if not phase2_success:
            self.logger.debug("_plan_to_contact_with_window: phase 2 (middle) failed")
            return False
        assert self._current_plan is not None
        middle_plan = self._resample_plan(self._current_plan, middle_steps + 1)
        phase2 = phase1.stack(middle_plan.trim_trajectory(1, None))

        # Phase 3: contact-allowed end window
        end_start = self._joint_state_from_plan_end(middle_plan)
        end_start_pose = self.get_ee_pose(end_start)
        phase3_success = self._plan_to_contact_pose(start_state=end_start, goal_pose=goal_pose, contact=True)
        self._log_phase_calculation(
            phase="exit",
            success=phase3_success,
            contact=True,
            start_position=self._pose_position_to_list(end_start_pose),
            goal_position=self._pose_position_to_list(goal_pose),
            allocated_steps=exit_steps,
        )
        if not phase3_success:
            self.logger.debug("_plan_to_contact_with_window: phase 3 (exit) failed")
            return False
        assert self._current_plan is not None
        exit_plan = self._resample_plan(self._current_plan, exit_steps + 1)

        self._current_plan = phase2.stack(exit_plan.trim_trajectory(1, None))
        self._plan_index = 0
        self._log_phase_calculation(
            phase="final",
            success=True,
            contact=False,
            start_position=self._pose_position_to_list(start_pose),
            goal_position=self._pose_position_to_list(goal_pose),
            final_waypoints=len(self._current_plan.position),
        )
        self.logger.info(f"_plan_to_contact_with_window succeeded: {len(self._current_plan.position)} waypoints")
        return True

    def _phase_step_budget(
        self,
        start_pose: Pose,
        goal_pose: Pose,
        contact_window_fraction: float,
    ) -> tuple[int, int, int]:
        """Split a distance-scaled waypoint budget across entry, middle, and exit phases."""
        total_steps = self._scaled_total_step_budget(start_pose=start_pose, goal_pose=goal_pose)
        entry_steps = max(2, int(round(total_steps * contact_window_fraction)))
        exit_steps = max(2, int(round(total_steps * contact_window_fraction)))
        middle_steps = total_steps - entry_steps - exit_steps

        if middle_steps < 2:
            deficit = 2 - middle_steps
            take_from_entry = min(deficit // 2 + deficit % 2, entry_steps - 2)
            entry_steps -= take_from_entry
            deficit -= take_from_entry
            take_from_exit = min(deficit, exit_steps - 2)
            exit_steps -= take_from_exit
            middle_steps = total_steps - entry_steps - exit_steps

        if middle_steps < 2:
            raise ValueError(
                "trajopt_tsteps is too small for three phases with minimum 2 steps each; "
                f"got trajopt_tsteps={self.config.trajopt_tsteps}"
            )

        return entry_steps, middle_steps, exit_steps

    def _scaled_total_step_budget(self, start_pose: Pose, goal_pose: Pose) -> int:
        """Scale trajopt_tsteps by the Cartesian span, using the task max span as the upper bound."""
        max_total_steps = max(int(self.config.trajopt_tsteps), 6)
        min_total_steps = min(6, max_total_steps)
        max_distance = float(self._MAX_TASK_TRAJECTORY_DISTANCE.item())
        distance = self._cartesian_distance(start_pose, goal_pose)

        if max_distance <= 0.0:
            return max_total_steps

        span_ratio = min(max(distance / max_distance, 0.0), 1.0)
        scaled_steps = min_total_steps + int(round((max_total_steps - min_total_steps) * span_ratio))
        return min(max(scaled_steps, min_total_steps), max_total_steps)

    def _log_phase_calculation(self, phase: str, success: bool, contact: bool, **extra_fields) -> None:
        """Append one phase event to a local JSONL debug file."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "success": success,
            "contact": contact,
        }
        payload.update(extra_fields)

        start_position = payload.get("start_position")
        goal_position = payload.get("goal_position")
        if isinstance(start_position, list) and isinstance(goal_position, list):
            payload["start_goal_distance"] = self._distance_between_positions(start_position, goal_position)

        attached_object = self._get_attached_object_log_data()
        if attached_object is not None:
            payload["attached_object"] = attached_object

        try:
            self._phase_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._phase_log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload) + "\n")
        except Exception as exc:
            self.logger.debug(f"Phase log write failed: {exc}")

    def _pose_position_to_list(self, pose: Pose) -> list[float]:
        """Convert Pose.position into a JSON-serializable xyz list."""
        assert pose.position is not None
        position = pose.position.squeeze(0) if pose.position.ndim > 1 else pose.position
        return [float(value) for value in position.tolist()]

    def _distance_between_positions(self, start_position: list[float], goal_position: list[float]) -> float:
        """Compute Euclidean distance between two xyz positions."""
        start_tensor = torch.tensor(start_position, dtype=torch.float32)
        goal_tensor = torch.tensor(goal_position, dtype=torch.float32)
        return float(torch.linalg.vector_norm(goal_tensor - start_tensor).item())

    def _get_attached_object_log_data(self) -> dict[str, object] | None:
        """Return attached object name and local position for phase logging."""
        attached_objects = self.get_attached_objects()
        if not attached_objects:
            return None

        object_name = attached_objects[0]
        rigid_objects = self.env.scene.rigid_objects
        if object_name not in rigid_objects:
            return {"name": object_name, "position": None}

        env_id = int(getattr(self, "env_id", 0))
        origin = self.env.scene.env_origins[env_id]
        object_position = rigid_objects[object_name].data.root_pos_w[env_id] - origin
        return {
            "name": object_name,
            "position": [float(value) for value in object_position.tolist()],
        }

    def _cartesian_distance(self, start_pose: Pose, goal_pose: Pose) -> float:
        """Compute the Cartesian distance between two EE poses."""
        assert start_pose.position is not None and goal_pose.position is not None
        start_position = start_pose.position.squeeze(0) if start_pose.position.ndim > 1 else start_pose.position
        goal_position = goal_pose.position.squeeze(0) if goal_pose.position.ndim > 1 else goal_pose.position
        return torch.linalg.vector_norm(goal_position - start_position).item()

    def _resample_plan(self, plan: JointState, num_steps: int) -> JointState:
        """Resample a trajectory to an exact number of waypoints using linear interpolation."""
        if num_steps < 2:
            raise ValueError(f"num_steps must be at least 2, got {num_steps}")
        if len(plan.position) == num_steps:
            return plan.clone()

        position = self._resample_tensor(cast(torch.Tensor, plan.position), num_steps)
        assert position is not None

        return JointState(
            position=position,
            velocity=self._resample_tensor(cast(torch.Tensor | None, plan.velocity), num_steps),
            acceleration=self._resample_tensor(cast(torch.Tensor | None, plan.acceleration), num_steps),
            joint_names=plan.joint_names,
            jerk=self._resample_tensor(cast(torch.Tensor | None, plan.jerk), num_steps),
            tensor_args=plan.tensor_args,
        )

    def _resample_tensor(self, values: torch.Tensor | None, num_steps: int) -> torch.Tensor | None:
        """Resample a [T, D] trajectory tensor along the time axis."""
        if values is None:
            return None
        if values.shape[0] == num_steps:
            return values.clone()

        interpolated = torch_f.interpolate(
            values.transpose(0, 1).unsqueeze(0),
            size=num_steps,
            mode="linear",
            align_corners=True,
        )
        return interpolated.squeeze(0).transpose(0, 1).contiguous()

    def _interpolate_pose(self, start_pose: Pose, goal_pose: Pose, fraction: float) -> Pose:
        """Interpolate position toward a goal pose while keeping the candidate goal orientation."""
        assert start_pose.position is not None and goal_pose.position is not None
        assert goal_pose.quaternion is not None
        start_position = start_pose.position.squeeze(0) if start_pose.position.ndim > 1 else start_pose.position
        goal_position = goal_pose.position.squeeze(0) if goal_pose.position.ndim > 1 else goal_pose.position
        position = start_position + (goal_position - start_position) * fraction
        quaternion = goal_pose.quaternion.squeeze(0) if goal_pose.quaternion.ndim > 1 else goal_pose.quaternion
        return self._make_pose(position=position, quaternion=quaternion)

    def _joint_state_from_plan_end(self, plan: JointState) -> JointState:
        """Build a cuRobo JointState from the final waypoint of a plan."""
        last_waypoint = torch.as_tensor(plan.position[-1], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        last_waypoint = last_waypoint.unsqueeze(0)
        current_state = JointState(
            position=last_waypoint,
            velocity=torch.zeros_like(last_waypoint),
            acceleration=torch.zeros_like(last_waypoint),
            joint_names=plan.joint_names,
        )
        return current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def _select_goal_index_from_plan(self, plan: JointState, candidate_poses: list[Pose]) -> int:
        """Infer which native goalset candidate was selected from the final EE pose."""
        final_state = self._joint_state_from_plan_end(plan)
        final_pose = self.get_ee_pose(final_state)
        assert final_pose.position is not None
        final_position = final_pose.position.squeeze(0) if final_pose.position.ndim > 1 else final_pose.position

        best_index = 0
        best_distance = float("inf")
        for idx, candidate_pose in enumerate(candidate_poses):
            assert candidate_pose.position is not None
            candidate_position = (
                candidate_pose.position.squeeze(0) if candidate_pose.position.ndim > 1 else candidate_pose.position
            )
            distance = torch.linalg.vector_norm(final_position - candidate_position).item()
            if distance < best_distance:
                best_distance = distance
                best_index = idx
        return best_index

    def _result_succeeded(self, result) -> bool:
        """Return a Python bool from a cuRobo result success field."""
        success = getattr(result, "success", False)
        if isinstance(success, torch.Tensor):
            try:
                return bool(success.item())
            except Exception:
                pass
        try:
            return bool(success)
        except Exception:
            return False

    def _extract_plan(self, result) -> JointState | None:
        """Extract JointState from MotionGen result."""
        # Try optimized_plan first
        if hasattr(result, "optimized_plan") and result.optimized_plan is not None:
            if hasattr(result.optimized_plan, "position") and len(result.optimized_plan.position) > 0:
                return result.optimized_plan
        # Some result objects expose the selected plan directly
        if hasattr(result, "plan") and result.plan is not None:
            if hasattr(result.plan, "position") and len(result.plan.position) > 0:
                return result.plan
        # Try get_interpolated_plan
        if hasattr(result, "get_interpolated_plan"):
            try:
                plan = result.get_interpolated_plan()
                if plan and hasattr(plan, "position") and len(plan.position) > 0:
                    return plan
            except Exception:
                pass
        # Return result if it's a JointState
        if isinstance(result, JointState) and hasattr(result, "position") and len(result.position) > 0:
            return result
        return None

    def _finalise_plan(self, plan: JointState) -> JointState:
        """Reorder plan to match robot joint ordering."""
        full = self.motion_gen.get_full_js(plan)
        if full.joint_names is None:
            return full
        common = [j for j in self.robot.data.joint_names if j in full.joint_names]
        return full.get_ordered_joint_state(common)
