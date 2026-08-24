#!/usr/bin/env python

import torch
from torch import nn

from lerobot.policies.act.modeling_act import ACT


class RecordingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        self.inputs.append(value.clone())
        return {"feature_map": value + 10}


def test_shared_backbone_batches_equal_camera_shapes_and_preserves_order() -> None:
    backbone = RecordingBackbone()
    images = [
        torch.full((2, 3, 4, 5), float(camera))
        for camera in range(3)
    ]

    features = ACT._extract_shared_camera_features(images, backbone)

    assert len(backbone.inputs) == 1
    assert backbone.inputs[0].shape == (6, 3, 4, 5)
    assert len(features) == 3
    for camera, feature in enumerate(features):
        torch.testing.assert_close(feature, images[camera] + 10)


def test_shared_backbone_keeps_single_camera_batch_unchanged() -> None:
    backbone = RecordingBackbone()
    image = torch.randn(4, 3, 4, 5)

    features = ACT._extract_shared_camera_features([image], backbone)

    assert len(backbone.inputs) == 1
    assert backbone.inputs[0].shape == image.shape
    torch.testing.assert_close(features[0], image + 10)


def test_shared_backbone_falls_back_for_different_camera_shapes() -> None:
    backbone = RecordingBackbone()
    images = [torch.randn(2, 3, 4, 5), torch.randn(2, 3, 5, 4)]

    features = ACT._extract_shared_camera_features(images, backbone)

    assert [value.shape for value in backbone.inputs] == [image.shape for image in images]
    assert [value.shape for value in features] == [image.shape for image in images]
