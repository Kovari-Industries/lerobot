from types import SimpleNamespace

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from torch import nn


class RecordingEncoder(nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = offset
        self.inputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.inputs.append(images.clone())
        return images[:, 0, 0, 0].unsqueeze(-1) + self.offset


def _model(*, separate: bool, encoders: nn.Module) -> DiffusionModel:
    model = DiffusionModel.__new__(DiffusionModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        image_features={"head": object(), "left": object(), "right": object()},
        use_separate_rgb_encoder_per_camera=separate,
        env_state_feature=None,
    )
    model.rgb_encoder = encoders
    return model


def _batch() -> dict[str, torch.Tensor]:
    images = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3, 1, 1, 1)
    return {
        OBS_STATE: torch.zeros(2, 2, 1),
        OBS_IMAGES: images,
    }


def test_diffusion_shared_encoder_remains_batched_across_all_cameras() -> None:
    encoder = RecordingEncoder()
    model = _model(separate=False, encoders=encoder)

    conditioning = model._prepare_global_conditioning(_batch()).reshape(2, 2, 4)

    assert [value.shape for value in encoder.inputs] == [(12, 1, 1, 1)]
    torch.testing.assert_close(conditioning[:, :, 1:], _batch()[OBS_IMAGES][..., 0, 0, 0])


def test_diffusion_separate_encoder_policy_remains_serial_per_camera() -> None:
    encoders = nn.ModuleList([RecordingEncoder(offset) for offset in (10.0, 20.0, 30.0)])
    model = _model(separate=True, encoders=encoders)

    conditioning = model._prepare_global_conditioning(_batch()).reshape(2, 2, 4)

    assert [[value.shape for value in encoder.inputs] for encoder in encoders] == [
        [(4, 1, 1, 1)],
        [(4, 1, 1, 1)],
        [(4, 1, 1, 1)],
    ]
    expected = _batch()[OBS_IMAGES][..., 0, 0, 0] + torch.tensor([10.0, 20.0, 30.0])
    torch.testing.assert_close(conditioning[:, :, 1:], expected)
