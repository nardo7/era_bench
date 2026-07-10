import torch

from isaaclab.managers.action_manager import ActionManager


class ActionManagerExtended(ActionManager):

    def process_action(self, action: torch.Tensor, action_names: list[str]):

        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for name in action_names:
            term = self._terms[name]
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self, action_names: list[str]):
        # apply the actions to the environment
        for name in action_names:
            term = self._terms[name]
            term.apply_actions()
