"""Extended cuRobo planner with plan_goalset support."""

from typing import Iterable

import torch
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
    ) -> bool:
        """Plan trajectory to the best reachable goal from a set of candidates.

        Uses cuRobo's batch goalset API if available (much faster), otherwise
        falls back to sequential per-goal planning.

        Args:
            positions: Iterable of 3-element position tensors (EE goal positions)
            quaternions: Iterable of 4-element quaternion tensors (w,x,y,z format)

        Returns:
            True if a trajectory was found, False otherwise.
        """
        # Build goal poses
        poses = [self._make_pose(position=pos, quaternion=quat) for pos, quat in zip(positions, quaternions)]
        if not poses:
            self.logger.warning("plan_goalset: empty goal set")
            return False

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
                        self.logger.info(f"plan_goalset succeeded: {len(self._current_plan.position)} waypoints")
                        return True
            except Exception as exc:
                self.logger.debug(f"{api_name} failed: {exc}")

        # Fallback: sequential per-goal planning
        self.logger.info("plan_goalset: falling back to sequential planning")
        for pose in poses:
            try:
                if self._plan_to_contact(
                    start_state=start_state,
                    goal_pose=pose,
                    retreat_distance=self.config.retreat_distance,
                    approach_distance=self.config.approach_distance,
                    contact=False,
                ):
                    return True
            except Exception:
                continue

        return False

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
