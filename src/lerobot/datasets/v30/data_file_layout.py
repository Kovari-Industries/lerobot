"""Dependency-free planning for LeRobot v3 data-file assignments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DataFileAssignment:
    chunk_index: int
    file_index: int


def plan_data_file_layout(
    episode_sizes_mb: Sequence[float],
    *,
    max_file_size_mb: float,
    files_per_chunk: int,
) -> tuple[DataFileAssignment, ...]:
    """Assign episodes before metadata or physical data files are written."""

    if not math.isfinite(max_file_size_mb) or max_file_size_mb <= 0:
        raise ValueError("max_file_size_mb must be positive and finite")
    if files_per_chunk <= 0:
        raise ValueError("files_per_chunk must be positive")

    assignments: list[DataFileAssignment] = []
    chunk_index = 0
    file_index = 0
    current_size_mb = 0.0
    current_file_has_episode = False
    for episode_size_mb in episode_sizes_mb:
        if not math.isfinite(episode_size_mb) or episode_size_mb < 0:
            raise ValueError("episode sizes must be finite and non-negative")
        if current_file_has_episode and current_size_mb + episode_size_mb >= max_file_size_mb:
            if file_index == files_per_chunk - 1:
                chunk_index += 1
                file_index = 0
            else:
                file_index += 1
            current_size_mb = 0.0
            current_file_has_episode = False
        assignments.append(
            DataFileAssignment(
                chunk_index=chunk_index,
                file_index=file_index,
            )
        )
        current_size_mb += episode_size_mb
        current_file_has_episode = True
    return tuple(assignments)
