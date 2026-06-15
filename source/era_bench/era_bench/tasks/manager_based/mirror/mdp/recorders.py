# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom recorder terms for capturing camera sensor data."""

from __future__ import annotations

import io
from collections.abc import Sequence

import imageio
import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg


class EpisodeVideoRecorder(RecorderTerm):
    """Recorder term that buffers RGB frames each step and writes one MP4 per episode.

    Frames are read directly from the TiledCamera sensor (raw uint8 0-255) rather than
    from obs_buf.  Using obs_buf with normalize=True would apply mean subtraction, which
    crushes colour information and produces desaturated videos.

    record_pre_step  — appends the current frame to an in-memory numpy buffer per env.
    record_pre_reset — encodes each resetting env's buffer to MP4 bytes stored in HDF5.
    reset            — clears the buffer for envs that are being reset.
    """

    def __init__(self, cfg: "EpisodeVideoRecorderCfg", env):
        super().__init__(cfg, env)
        self._sensor_name = cfg.sensor_name
        self._fps = cfg.fps

        # per-env frame buffer (numpy array, lazily allocated) and episode counter
        self._buffers: dict[int, np.ndarray | None] = {i: None for i in range(env.num_envs)}
        self._ep_counts: dict[int, int] = {i: 0 for i in range(env.num_envs)}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = range(self._env.num_envs)
        for env_id in env_ids:
            self._buffers[env_id] = None

    def record_pre_step(self):
        """Buffer one raw uint8 RGB frame per env directly from the camera sensor.

        Reading from the sensor bypasses any obs normalization (e.g. mean subtraction)
        that would otherwise destroy colour information in the video.
        """
        sensor = self._env.scene.sensors.get(self._sensor_name)
        if sensor is None:
            return None, None

        # TiledCamera output is uint8 RGBA or RGB, shape (N, H, W, C)
        frames = sensor.data.output["rgb"].cpu().numpy()  # uint8, no normalization
        for env_id in range(self._env.num_envs):
            frame = frames[env_id][np.newaxis]  # (1, H, W, C)
            existing = self._buffers[env_id]
            if existing is None:
                self._buffers[env_id] = frame
            else:
                self._buffers[env_id] = np.concatenate([existing, frame], axis=0)

        return None, None  # nothing to store in HDF5 at this point

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        """Encode frames to MP4 bytes and return them to be stored in the HDF5 episode.

        Returns a Python list of 1-D uint8 tensors (one per env in env_ids) so that each
        env's video can have a different byte length.  add_to_episodes only does
        value[value_index] indexing, so a list works just as well as a stacked tensor.
        Envs whose buffer is still empty (e.g. on the very first reset) receive an empty
        placeholder tensor (length 0) so the index mapping stays intact.
        """
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        video_list: list[torch.Tensor] = []
        any_valid = False

        for env_id in env_ids:
            frames_np = self._buffers[env_id]  # (T, H, W, C) or None

            if frames_np is None:
                # No frames yet for this env (very first reset); store an empty placeholder
                # so the index alignment with env_ids is preserved.
                video_list.append(torch.zeros(0, dtype=torch.uint8))
                continue

            any_valid = True

            # float [0,1] -> uint8
            if frames_np.dtype != np.uint8:
                frames_np = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
            # drop alpha channel if RGBA
            if frames_np.shape[-1] == 4:
                frames_np = frames_np[..., :3]

            # encode MP4 into an in-memory buffer
            buf = io.BytesIO()
            writer = imageio.get_writer(buf, format="mp4", fps=self._fps, codec="libx264", quality=8)
            for frame in frames_np:
                writer.append_data(frame)
            writer.close()
            mp4_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # 1-D uint8

            print(
                f"[EpisodeVideoRecorder] env{env_id} ep{self._ep_counts[env_id]}: "
                f"{frames_np.shape} -> {mp4_bytes.nbytes / 1024:.1f} KB stored in HDF5"
            )

            self._ep_counts[env_id] += 1
            self._buffers[env_id] = None
            video_list.append(torch.from_numpy(mp4_bytes))

        if not any_valid:
            return None, None  # very first reset — nothing buffered yet

        return "video", video_list


@configclass
class EpisodeVideoRecorderCfg(RecorderTermCfg):
    """Configuration for the episode video recorder term."""

    class_type: type[RecorderTerm] = EpisodeVideoRecorder

    sensor_name: str = "tiled_camera"
    """Name of the TiledCamera sensor in the scene (must match SceneCfg attribute name)."""

    fps: int = 20
    """Frames per second of the encoded MP4."""


@configclass
class RecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Extends the default recorder with per-episode video generation."""

    record_episode_video = EpisodeVideoRecorderCfg()
