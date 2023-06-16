from typing import Optional, Sequence, cast

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.torch.policies import Policy
from ...models.torch.q_functions.ensemble_q_function import EnsembleQFunction
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import (
    eval_api,
    freeze,
    get_state_dict,
    hard_sync,
    map_location,
    reset_optimizer_states,
    set_state_dict,
    sync_optimizer_state,
    to_cpu,
    to_cuda,
    torch_api,
    unfreeze,
)
from ..base import AlgoImplBase


class TorchImplBase(AlgoImplBase):

    _observation_shape: Sequence[int]
    _action_size: int
    _scaler: Optional[Scaler]
    _action_scaler: Optional[ActionScaler]
    _reward_scaler: Optional[RewardScaler]
    _device: str

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._scaler = scaler
        self._action_scaler = action_scaler
        self._reward_scaler = reward_scaler
        self._device = "cpu:0"

    @eval_api
    @torch_api(scaler_targets=["x"])
    def predict_best_action(self, x: torch.Tensor) -> np.ndarray:
        assert x.ndim > 1, "Input must have batch dimension."

        with torch.no_grad():
            action = self._predict_best_action(x)

            # transform action back to the original range
            if self._action_scaler:
                action = self._action_scaler.reverse_transform(action)

            return action.cpu().detach().numpy()

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @eval_api
    @torch_api(scaler_targets=["x"])
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        assert x.ndim > 1, "Input must have batch dimension."

        with torch.no_grad():
            action = self._sample_action(x)

            # transform action back to the original range
            if self._action_scaler:
                action = self._action_scaler.reverse_transform(action)

            return action.cpu().detach().numpy()

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @eval_api
    def save_policy(self, fname: str) -> None:
        dummy_x = torch.rand(1, *self.observation_shape, device=self._device)

        # workaround until version 1.6
        freeze(self)

        # dummy function to select best actions
        def _func(x: torch.Tensor) -> torch.Tensor:
            if self._scaler:
                x = self._scaler.transform(x)

            action = self._predict_best_action(x)

            if self._action_scaler:
                action = self._action_scaler.reverse_transform(action)

            return action

        traced_script = torch.jit.trace(_func, dummy_x, check_trace=False)

        if fname.endswith(".onnx"):
            # currently, PyTorch cannot directly export function as ONNX.
            torch.onnx.export(
                traced_script,
                dummy_x,
                fname,
                export_params=True,
                opset_version=11,
                input_names=["input_0"],
                output_names=["output_0"],
                example_outputs=traced_script(dummy_x),
            )
        elif fname.endswith(".pt"):
            traced_script.save(fname)
        else:
            raise ValueError(
                f"invalid format type: {fname}."
                " .pt and .onnx extensions are currently supported."
            )

        # workaround until version 1.6
        unfreeze(self)

    def to_gpu(self, device: Device = Device()) -> None:
        self._device = f"cuda:{device.get_id()}"
        to_cuda(self, self._device)

    def to_cpu(self) -> None:
        self._device = "cpu:0"
        to_cpu(self)

    def save_model(self, fname: str) -> None:
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=map_location(self._device))
        set_state_dict(self, chkpt)

    @property
    def policy(self) -> Policy:
        raise NotImplementedError

    def copy_policy_from(self, impl: AlgoImplBase) -> None:
        impl = cast("TorchImplBase", impl)
        if not isinstance(impl.policy, type(self.policy)):
            raise ValueError(
                f"Invalid policy type: expected={type(self.policy)},"
                f"actual={type(impl.policy)}"
            )
        hard_sync(self.policy, impl.policy)

    @property
    def policy_optim(self) -> Optimizer:
        raise NotImplementedError

    def copy_policy_optim_from(self, impl: AlgoImplBase) -> None:
        impl = cast("TorchImplBase", impl)
        if not isinstance(impl.policy_optim, type(self.policy_optim)):
            raise ValueError(
                "Invalid policy optimizer type: "
                f"expected={type(self.policy_optim)},"
                f"actual={type(impl.policy_optim)}"
            )
        sync_optimizer_state(self.policy_optim, impl.policy_optim)

    @property
    def q_function(self) -> EnsembleQFunction:
        raise NotImplementedError

    def copy_q_function_from(self, impl: AlgoImplBase) -> None:
        impl = cast("TorchImplBase", impl)
        q_func = self.q_function.q_funcs[0]
        if not isinstance(impl.q_function.q_funcs[0], type(q_func)):
            raise ValueError(
                f"Invalid Q-function type: expected={type(q_func)},"
                f"actual={type(impl.q_function.q_funcs[0])}"
            )
        hard_sync(self.q_function, impl.q_function)

    @property
    def q_function_optim(self) -> Optimizer:
        raise NotImplementedError

    def copy_q_function_optim_from(self, impl: AlgoImplBase) -> None:
        impl = cast("TorchImplBase", impl)
        if not isinstance(impl.q_function_optim, type(self.q_function_optim)):
            raise ValueError(
                "Invalid Q-function optimizer type: "
                f"expected={type(self.q_function_optim)}",
                f"actual={type(impl.q_function_optim)}",
            )
        sync_optimizer_state(self.q_function_optim, impl.q_function_optim)

    def reset_optimizer_states(self) -> None:
        reset_optimizer_states(self)

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def scaler(self) -> Optional[Scaler]:
        return self._scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        return self._action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        return self._reward_scaler
