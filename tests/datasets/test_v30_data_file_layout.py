from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.datasets.v30.data_file_layout import DataFileAssignment, plan_data_file_layout


def assignment_pairs(assignments: tuple[DataFileAssignment, ...]) -> list[tuple[int, int]]:
    return [(assignment.chunk_index, assignment.file_index) for assignment in assignments]


@pytest.mark.parametrize(
    ("sizes", "limit", "expected"),
    [
        ([], 100.0, []),
        ([40.0, 40.0], 100.0, [(0, 0), (0, 0)]),
        ([50.0, 50.0], 100.0, [(0, 0), (0, 1)]),
        ([150.0], 100.0, [(0, 0)]),
        ([150.0, 150.0], 100.0, [(0, 0), (0, 1)]),
        ([150.0, 40.0, 40.0], 100.0, [(0, 0), (0, 1), (0, 1)]),
    ],
)
def test_plan_data_file_layout(
    sizes: list[float],
    limit: float,
    expected: list[tuple[int, int]],
) -> None:
    assignments = plan_data_file_layout(
        sizes,
        max_file_size_mb=limit,
        files_per_chunk=1000,
    )

    assert assignment_pairs(assignments) == expected


def test_plan_data_file_layout_rolls_over_chunks() -> None:
    assignments = plan_data_file_layout(
        [150.0, 150.0, 150.0],
        max_file_size_mb=100.0,
        files_per_chunk=2,
    )

    assert assignment_pairs(assignments) == [(0, 0), (0, 1), (1, 0)]


@pytest.mark.parametrize(
    ("sizes", "limit", "files_per_chunk"),
    [
        ([-1.0], 100.0, 1000),
        ([float("nan")], 100.0, 1000),
        ([1.0], 0.0, 1000),
        ([1.0], float("inf"), 1000),
        ([1.0], 100.0, 0),
    ],
)
def test_plan_data_file_layout_rejects_invalid_inputs(
    sizes: list[float],
    limit: float,
    files_per_chunk: int,
) -> None:
    with pytest.raises(ValueError):
        plan_data_file_layout(
            sizes,
            max_file_size_mb=limit,
            files_per_chunk=files_per_chunk,
        )


def test_convert_data_metadata_matches_physical_file_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torchvision")
    from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as converter

    root = tmp_path / "source"
    data_dir = root / "data/chunk-000"
    data_dir.mkdir(parents=True)
    episode_paths = [data_dir / f"episode_{episode_index:06d}.parquet" for episode_index in range(3)]
    for episode_path in episode_paths:
        episode_path.touch()

    sizes_by_path = dict(zip(episode_paths, [150.0, 40.0, 40.0], strict=True))
    frames_by_path = dict(zip(episode_paths, [3, 4, 5], strict=True))
    writes: list[tuple[int, int, tuple[Path, ...]]] = []

    monkeypatch.setattr(converter, "get_image_keys", lambda _root: [])
    monkeypatch.setattr(
        converter,
        "get_parquet_file_size_in_mb",
        lambda path: sizes_by_path[path],
    )
    monkeypatch.setattr(
        converter,
        "get_parquet_num_frames",
        lambda path: frames_by_path[path],
    )
    monkeypatch.setattr(
        converter,
        "concat_data_files",
        lambda paths, _new_root, chunk_index, file_index, _image_keys: writes.append(
            (chunk_index, file_index, tuple(paths))
        ),
    )

    metadata = converter.convert_data(
        root,
        tmp_path / "converted",
        data_file_size_in_mb=100,
    )

    physical_assignment = {
        path: (chunk_index, file_index) for chunk_index, file_index, paths in writes for path in paths
    }
    declared_assignment = [(row["data/chunk_index"], row["data/file_index"]) for row in metadata]

    assert declared_assignment == [physical_assignment[path] for path in episode_paths]
    assert declared_assignment == [(0, 0), (0, 1), (0, 1)]
    assert [(row["dataset_from_index"], row["dataset_to_index"]) for row in metadata] == [
        (0, 3),
        (3, 7),
        (7, 12),
    ]
