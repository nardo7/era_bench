"""Extended cuRobo planner with plan_goalset support."""

from typing import Iterable, cast

import torch
import torch.nn.functional as torch_f
from curobo.types.math import Pose
from curobo.types.state import JointState
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner


class ExtendedCuroboPlanner(CuroboPlanner):
    """CuroboPlanner subclass that adds plan_goalset for batch goal evaluation.

    Attempts to use cuRobo's native plan_goalset / plan_batch_goalset APIs when
    available. Falls back to sequential per-goal planning for safety.
    """

    def plan_goalset(
        self,
        positions: Iterable[torch.Tensor],
        quaternions: Iterable[torch.Tensor],
        contact_window_fraction: float | None = None,
    ) -> bool:
        """Plan trajectory to the best reachable goal from a set of candidates.

        Tries cuRobo's native goalset API first. On failure falls back to sequential
        per-goal planning, using contact-window multi-phase planning when
        contact_window_fraction is provided.

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
        poses = [self._make_pose(position=pos, quaternion=quat) for pos, quat in zip(positions, quaternions)]
        if not poses:
            self.logger.warning("plan_goalset: empty goal set")
            return False

        self.reset_plan()
        self.update_world()
        start_state = self._get_current_joint_state_for_curobo()

        # Try native cuRobo goalset API
        mg = self.motion_gen
        for api_name in ("plan_goalset", "plan_batch_goalset"):
            fn = getattr(mg, api_name, None)
            if fn is None:
                continue
            try:
                result = fn(start_state, poses, self.plan_config)
                if result is not None and self._result_succeeded(result):
                    plan = self._extract_plan(result)
                    if plan is not None:
                        self._current_plan = self._finalise_plan(plan)
                        self._plan_index = 0
                        self.logger.info(f"plan_goalset native succeeded: {len(self._current_plan.position)} waypoints")
                        return True
            except Exception as exc:
                self.logger.debug(f"{api_name} failed: {exc}")

        # Fallback: sequential per-goal planning
        self.logger.info("plan_goalset: native failed, falling back to sequential planning")
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
        entry_steps, middle_steps, exit_steps = self._phase_step_budget(contact_window_fraction)

        # Phase 1: contact-allowed start window
        if not self._plan_to_contact_pose(start_state=start_state, goal_pose=entry_pose, contact=True):
            self.logger.debug("_plan_to_contact_with_window: phase 1 (entry) failed")
            return False
        assert self._current_plan is not None
        phase1 = self._resample_plan(self._current_plan, entry_steps)

        # Phase 2: collision-checked middle
        mid_start = self._joint_state_from_plan_end(phase1)
        if not self._plan_to_contact_pose(start_state=mid_start, goal_pose=exit_pose, contact=False):
            self.logger.debug("_plan_to_contact_with_window: phase 2 (middle) failed")
            return False
        assert self._current_plan is not None
        middle_plan = self._resample_plan(self._current_plan, middle_steps + 1)
        phase2 = phase1.stack(middle_plan.trim_trajectory(1, None))

        # Phase 3: contact-allowed end window
        end_start = self._joint_state_from_plan_end(middle_plan)
        if not self._plan_to_contact_pose(start_state=end_start, goal_pose=goal_pose, contact=True):
            self.logger.debug("_plan_to_contact_with_window: phase 3 (exit) failed")
            return False
        assert self._current_plan is not None
        exit_plan = self._resample_plan(self._current_plan, exit_steps + 1)

        self._current_plan = phase2.stack(exit_plan.trim_trajectory(1, None))
        self._plan_index = 0
        self.logger.info(f"_plan_to_contact_with_window succeeded: {len(self._current_plan.position)} waypoints")
        return True

    def _phase_step_budget(self, contact_window_fraction: float) -> tuple[int, int, int]:
        """Split the configured total waypoint budget across entry, middle, and exit phases."""
        total_steps = max(int(self.config.trajopt_tsteps), 6)
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
