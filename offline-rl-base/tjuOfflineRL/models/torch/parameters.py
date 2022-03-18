import torch
from torch import nn


class Parameter(nn.Module):  # type: ignore

    _parameter: nn.Parameter

    def __init__(self, data: torch.Tensor):
        super().__init__()
        self._parameter = nn.Parameter(data)

    def forward(self) -> torch.Tensor:
        return self._parameter

    def __call__(self) -> torch.Tensor:
        return super().__call__()

    @property
    def data(self) -> torch.Tensor:
        return self._parameter.data
