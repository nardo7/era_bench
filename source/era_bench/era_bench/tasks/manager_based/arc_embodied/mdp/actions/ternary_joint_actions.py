"""A three-position joint action for the Franka parallel gripper."""

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
import isaaclab.utils.string as string_utils


class TernaryJointPositionAction(BinaryJointPositionAction):
    """Map a scalar action to closed, release-open, or wide-open targets.

    A value below ``-0.5`` closes the gripper, a value in ``[-0.5, 0.5]``
    selects the narrow release aperture, and a value above ``0.5`` selects
    the wide-open aperture.
    """

    cfg: "TernaryJointPositionActionCfg"

    def __init__(self, cfg: "TernaryJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)

        self._release_command = torch.zeros(self._num_joints, device=self.device)
        indices, names, values = string_utils.resolve_matching_names_values(
            cfg.release_command_expr, self._joint_names
        )
        if len(indices) != self._num_joints:
            missing = set(self._joint_names) - set(names)
            raise ValueError(f"Could not resolve release commands for joints: {missing}")
        self._release_command[indices] = torch.tensor(values, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        close_mask = actions < -0.5
        wide_open_mask = actions > 0.5
        self._processed_actions = torch.where(
            close_mask,
            self._close_command,
            torch.where(wide_open_mask, self._open_command, self._release_command),
        )

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class TernaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for :class:`TernaryJointPositionAction`."""

    release_command_expr: dict[str, float] = MISSING
    """The joint command for the narrow release aperture."""

    class_type: type[ActionTerm] = TernaryJointPositionAction
