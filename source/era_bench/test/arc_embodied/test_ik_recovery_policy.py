from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _ensure_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(package_path)]
    sys.modules[name] = module


def _load_module(name: str, module_path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "era_bench"
_ensure_package("era_bench", PACKAGE_ROOT)
_ensure_package("era_bench.tasks", PACKAGE_ROOT / "tasks")
_ensure_package("era_bench.tasks.manager_based", PACKAGE_ROOT / "tasks" / "manager_based")
_ensure_package(
    "era_bench.tasks.manager_based.common",
    PACKAGE_ROOT / "tasks" / "manager_based" / "common",
)
_ensure_package(
    "era_bench.tasks.manager_based.arc_embodied",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied",
)
_ensure_package(
    "era_bench.tasks.manager_based.arc_embodied.control",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied" / "control",
)

progress_watchdog_module = _load_module(
    "era_bench.tasks.manager_based.common.progress_watchdog",
    PACKAGE_ROOT / "tasks" / "manager_based" / "common" / "progress_watchdog.py",
)
arc_state_machine_module = _load_module(
    "era_bench.tasks.manager_based.arc_embodied.control.state_machine",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied" / "control" / "state_machine.py",
)
ik_watchdog_module = _load_module(
    "era_bench.tasks.manager_based.arc_embodied.control.ik_watchdog",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied" / "control" / "ik_watchdog.py",
)
ik_recovery_policy_module = _load_module(
    "era_bench.tasks.manager_based.arc_embodied.control.ik_recovery",
    PACKAGE_ROOT / "tasks" / "manager_based" / "arc_embodied" / "control" / "ik_recovery.py",
)

ArcStates = arc_state_machine_module.ArcStates
ProgressWatchdogResult = progress_watchdog_module.ProgressWatchdogResult
ArcIKRecoveryPolicy = ik_recovery_policy_module.ArcIKRecoveryPolicy
ArcIKWatchdog = ik_watchdog_module.ArcIKWatchdog
ArcIKWatchdogUpdate = ik_watchdog_module.ArcIKWatchdogUpdate


def _watchdog_update(*warning_env_ids: int) -> ArcIKWatchdogUpdate:
    env_ids = torch.tensor(warning_env_ids, dtype=torch.long)
    active_mask = torch.zeros(1, dtype=torch.bool)
    if env_ids.numel() > 0:
        active_mask[env_ids] = True
    return ArcIKWatchdogUpdate(
        warning_messages=tuple(),
        warning_env_ids=env_ids,
        active_mask=active_mask,
        watchdog_result=ProgressWatchdogResult(
            error=torch.zeros(1, dtype=torch.float32),
            improved=torch.zeros(1, dtype=torch.bool),
            stalled=torch.zeros(1, dtype=torch.bool),
            diverged=torch.zeros(1, dtype=torch.bool),
            stagnant_steps=torch.zeros(1, dtype=torch.int64),
            best_error=torch.zeros(1, dtype=torch.float32),
        ),
    )


def test_recovery_policy_accepts_multiple_watched_states(tmp_path) -> None:
    policy = ArcIKRecoveryPolicy(
        num_envs=1,
        device="cpu",
        watched_states=(ArcStates.APPROACHING_PLACING_ABOVE, ArcStates.RETREATING_PLACING_ABOVE),
        log_path=str(tmp_path / "arc_ik_recovery.log"),
    )
    tcp_position = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    tcp_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    target_command = torch.tensor([[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=torch.float32)

    started = policy.update(
        arc_state=torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value]),
        tcp_position=tcp_position,
        tcp_orientation=tcp_orientation,
        target_command=target_command,
        watchdog_update=_watchdog_update(0),
    )
    assert started.active_mask.tolist() == [True]
    assert started.watchdog_reset_env_ids.tolist() == [0]
    assert started.request_curobo_env_ids.numel() == 0

    still_active = policy.update(
        arc_state=torch.tensor([ArcStates.RETREATING_PLACING_ABOVE.python_value]),
        tcp_position=tcp_position,
        tcp_orientation=tcp_orientation,
        target_command=target_command,
        watchdog_update=_watchdog_update(),
    )
    assert still_active.active_mask.tolist() == [True]
    assert still_active.watchdog_reset_env_ids.tolist() == [0]
    assert still_active.request_curobo_env_ids.numel() == 0

    reset = policy.update(
        arc_state=torch.tensor([ArcStates.REST.python_value]),
        tcp_position=tcp_position,
        tcp_orientation=tcp_orientation,
        target_command=target_command,
        watchdog_update=_watchdog_update(),
    )
    assert reset.active_mask.tolist() == [False]
    assert reset.watchdog_reset_env_ids.tolist() == [0]
    assert reset.request_curobo_env_ids.numel() == 0


def test_recovery_policy_logs_watchdog_reason(tmp_path) -> None:
    policy = ArcIKRecoveryPolicy(
        num_envs=1,
        device="cpu",
        watched_states=ArcStates.APPROACHING_PLACING_ABOVE,
        log_path=str(tmp_path / "arc_ik_recovery.log"),
    )
    watchdog_update = ArcIKWatchdogUpdate(
        warning_messages=(
            "[WARNING]: Arc IK watchdog detected stall env=0 state=APPROACHING_PLACING_ABOVE "
            "error=0.0588 best_error=0.0600 stagnant_steps=3 stalled=True diverged=False",
        ),
        warning_env_ids=torch.tensor([0], dtype=torch.long),
        active_mask=torch.tensor([True]),
        watchdog_result=ProgressWatchdogResult(
            error=torch.tensor([0.0588], dtype=torch.float32),
            improved=torch.tensor([False]),
            stalled=torch.tensor([True]),
            diverged=torch.tensor([False]),
            stagnant_steps=torch.tensor([3], dtype=torch.int64),
            best_error=torch.tensor([0.0600], dtype=torch.float32),
        ),
    )

    update = policy.update(
        arc_state=torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value]),
        tcp_position=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        tcp_orientation=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        target_command=torch.tensor([[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=torch.float32),
        watchdog_update=watchdog_update,
        step=43,
        elapsed_s=5.23,
    )

    assert any("detected stall" in message for message in update.messages)
    assert any("trigger=stall" in message for message in update.messages)
    assert any("escalating to curobo" not in message for message in update.messages)


def test_watchdog_warns_on_slow_convergence_stall() -> None:
    watchdog = ArcIKWatchdog(
        num_envs=1,
        device="cpu",
        patience=3,
        warmup_steps=0,
        min_improvement=1e-3,
        divergence_ratio=10.0,
        divergence_abs_delta=10.0,
    )
    arc_state = torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value], dtype=torch.int64)
    target_pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    tcp_positions = [
        torch.tensor([[0.0600, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0595, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0591, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0588, 0.0, 0.0]], dtype=torch.float32),
    ]

    updates = [watchdog.update(arc_state, tcp_position, target_pose) for tcp_position in tcp_positions]

    assert updates[-1].watchdog_result.stalled.tolist() == [True]
    assert updates[-1].warning_env_ids.tolist() == [0]
    assert "detected stall" in updates[-1].warning_messages[0]


def test_watchdog_does_not_seed_zero_best_error_while_inactive() -> None:
    watchdog = ArcIKWatchdog(
        num_envs=1,
        device="cpu",
        patience=10,
        warmup_steps=3,
        min_improvement=1e-3,
        divergence_ratio=1.25,
        divergence_abs_delta=0.05,
    )
    inactive_state = torch.tensor([ArcStates.REST.python_value], dtype=torch.int64)
    active_state = torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value], dtype=torch.int64)
    tcp_position = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    target_pose = torch.tensor([[0.6277, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    inactive_update = watchdog.update(inactive_state, tcp_position, target_pose)
    assert torch.isinf(inactive_update.watchdog_result.best_error).tolist() == [True]
    assert inactive_update.warning_env_ids.numel() == 0

    active_update = watchdog.update(active_state, tcp_position, target_pose)
    assert active_update.watchdog_result.best_error.tolist() == [0.6276999711990356]
    assert active_update.watchdog_result.diverged.tolist() == [False]
    assert active_update.warning_env_ids.numel() == 0


def test_watchdog_requires_two_consecutive_divergence_updates() -> None:
    watchdog = ArcIKWatchdog(
        num_envs=1,
        device="cpu",
        patience=10,
        divergence_patience=2,
        warmup_steps=0,
        min_improvement=1e-3,
        divergence_ratio=1.25,
        divergence_abs_delta=0.05,
    )
    arc_state = torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value], dtype=torch.int64)
    seed_target_pose = torch.tensor([[0.0100, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    diverging_target_pose = torch.tensor([[0.2000, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    tcp_position = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    seeded_update = watchdog.update(arc_state, tcp_position, seed_target_pose)
    first_divergent_update = watchdog.update(arc_state, tcp_position, diverging_target_pose)
    second_divergent_update = watchdog.update(arc_state, tcp_position, diverging_target_pose)

    assert seeded_update.warning_env_ids.numel() == 0
    assert first_divergent_update.watchdog_result.diverged.tolist() == [True]
    assert first_divergent_update.warning_env_ids.numel() == 0
    assert second_divergent_update.watchdog_result.diverged.tolist() == [True]
    assert second_divergent_update.warning_env_ids.tolist() == [0]
    assert "detected divergence" in second_divergent_update.warning_messages[0]


def test_retreat_height_grows_with_retry_attempt(tmp_path) -> None:
    policy = ArcIKRecoveryPolicy(
        num_envs=1,
        device="cpu",
        retreat_height=0.12,
        retreat_height_step=0.03,
        retreat_lateral_radius=0.02,
        retreat_lateral_radius_step=0.01,
        log_path=str(tmp_path / "arc_ik_recovery.log"),
    )
    tcp_position = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    tcp_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    target_command = torch.tensor([[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=torch.float32)

    policy._attempt_idx[0] = 0
    policy._start_retreat(0, tcp_position, tcp_orientation, target_command)
    first_retreat_xy = policy._retreat_command[0, :2].clone()
    first_retreat_z = policy._retreat_command[0, 2].item()

    policy._attempt_idx[0] = 1
    policy._start_retreat(0, tcp_position, tcp_orientation, target_command)
    second_retreat_xy = policy._retreat_command[0, :2].clone()
    second_retreat_z = policy._retreat_command[0, 2].item()

    assert torch.linalg.vector_norm(first_retreat_xy - tcp_position[0, :2]).item() == pytest.approx(0.02)
    assert torch.linalg.vector_norm(second_retreat_xy - tcp_position[0, :2]).item() == pytest.approx(0.03)
    assert first_retreat_z == pytest.approx(0.42)
    assert second_retreat_z == pytest.approx(0.45)


def test_recovery_policy_requests_curobo_after_first_retreat(tmp_path) -> None:
    policy = ArcIKRecoveryPolicy(
        num_envs=1,
        device="cpu",
        retreat_height=0.08,
        retreat_height_step=0.02,
        retreat_lateral_radius=0.03,
        retreat_lateral_radius_step=0.02,
        max_retreat_steps=1,
        log_path=str(tmp_path / "arc_ik_recovery.log"),
    )
    tcp_position = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    tcp_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    target_command = torch.tensor([[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=torch.float32)

    started = policy.update(
        arc_state=torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value]),
        tcp_position=tcp_position,
        tcp_orientation=tcp_orientation,
        target_command=target_command,
        watchdog_update=_watchdog_update(0),
    )
    assert started.watchdog_reset_env_ids.tolist() == [0]
    assert started.request_curobo_env_ids.numel() == 0

    escalated = policy.update(
        arc_state=torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value]),
        tcp_position=tcp_position,
        tcp_orientation=tcp_orientation,
        target_command=target_command,
        watchdog_update=_watchdog_update(),
    )
    assert escalated.watchdog_reset_env_ids.tolist() == [0]
    assert escalated.request_curobo_env_ids.tolist() == [0]
    assert any("requesting curobo" in message for message in escalated.messages)


def test_recovery_policy_leaves_planner_failure_to_hybrid_action(tmp_path) -> None:
    policy = ArcIKRecoveryPolicy(
        num_envs=1,
        device="cpu",
        max_retreat_steps=1,
        log_path=str(tmp_path / "arc_ik_recovery.log"),
    )
    tcp_position = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    tcp_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    target_command = torch.tensor([[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=torch.float32)

    policy.update(
        arc_state=torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value]),
        tcp_position=tcp_position,
        tcp_orientation=tcp_orientation,
        target_command=target_command,
        watchdog_update=_watchdog_update(0),
    )

    failed = policy.update(
        arc_state=torch.tensor([ArcStates.APPROACHING_PLACING_ABOVE.python_value]),
        tcp_position=tcp_position,
        tcp_orientation=tcp_orientation,
        target_command=target_command,
        watchdog_update=_watchdog_update(),
    )

    assert failed.request_curobo_env_ids.tolist() == [0]
    assert any("requesting curobo" in message for message in failed.messages)
