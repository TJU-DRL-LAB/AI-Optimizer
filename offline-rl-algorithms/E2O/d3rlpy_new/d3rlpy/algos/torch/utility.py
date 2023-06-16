from typing import Optional, Tuple, Union

import numpy as np
import torch
from typing_extensions import Protocol

from ...models.torch import (
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
)
from ...torch_utility import eval_api, torch_api


class _DiscreteQFunctionProtocol(Protocol):
    _q_func: Optional[EnsembleDiscreteQFunction]


class _ContinuousQFunctionProtocol(Protocol):
    _q_func: Optional[EnsembleContinuousQFunction]


class DiscreteQFunctionMixin:
    @eval_api
    @torch_api(scaler_targets=["x"])
    def predict_value(
        self: _DiscreteQFunctionProtocol,
        x: torch.Tensor,
        action: torch.Tensor,
        with_std: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert x.ndim > 1, "Input must have batch dimension."
        assert x.shape[0] == action.shape[0]
        assert self._q_func is not None

        action = action.view(-1).long().cpu().detach().numpy()
        with torch.no_grad():
            values = self._q_func(x, reduction="none").cpu().detach().numpy()
            values = np.transpose(values, [1, 0, 2])

        mean_values = values.mean(axis=1)
        stds = np.std(values, axis=1)

        ret_values = []
        ret_stds = []
        for v, std, a in zip(mean_values, stds, action):
            ret_values.append(v[a])
            ret_stds.append(std[a])

        if with_std:
            return np.array(ret_values), np.array(ret_stds)

        return np.array(ret_values)


class ContinuousQFunctionMixin:
    @eval_api
    @torch_api(scaler_targets=["x"], action_scaler_targets=["action"])
    def predict_value(
        self: _ContinuousQFunctionProtocol,
        x: torch.Tensor,
        action: torch.Tensor,
        with_std: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert x.ndim > 1, "Input must have batch dimension."
        assert x.shape[0] == action.shape[0]
        assert self._q_func is not None

        with torch.no_grad():
            values = self._q_func(x, action, "none").cpu().detach().numpy()
            values = np.transpose(values, [1, 0, 2])

        mean_values = values.mean(axis=1).reshape(-1)
        stds = np.std(values, axis=1).reshape(-1)

        if with_std:
            return mean_values, stds

        return mean_values
