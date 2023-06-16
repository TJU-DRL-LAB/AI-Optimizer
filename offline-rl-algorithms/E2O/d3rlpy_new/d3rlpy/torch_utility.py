import collections
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data._utils.collate import default_collate
from typing_extensions import Protocol

from d3rlpy.dataset import TransitionMiniBatch
from .preprocessing import ActionScaler, RewardScaler, Scaler

BLACK_LIST = [
    "policy",
    "q_function",
    "policy_optim",
    "q_function_optim",
]  # special properties


def _get_attributes(obj: Any) -> List[str]:
    return [key for key in dir(obj) if key not in BLACK_LIST]


def soft_sync(targ_model: nn.Module, model: nn.Module, tau: float) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)


def soft_sync_ensemble(targ_model: List[nn.Module], model: List[nn.Module], tau: float, num_ensemble: int) -> None:
    for i in range(num_ensemble):
        with torch.no_grad():
            params = model[i].parameters()
            targ_params = targ_model[i].parameters()
            for p, p_targ in zip(params, targ_params):
                p_targ.data.mul_(1 - tau)
                p_targ.data.add_(tau * p.data)


def hard_sync(targ_model: nn.Module, model: nn.Module) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.copy_(p.data)


def sync_optimizer_state(targ_optim: Optimizer, optim: Optimizer) -> None:
    # source optimizer state
    state = optim.state_dict()["state"]
    # destination optimizer param_groups
    param_groups = targ_optim.state_dict()["param_groups"]
    # update only state
    targ_optim.load_state_dict({"state": state, "param_groups": param_groups})


def set_eval_mode(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.eval()


def set_train_mode(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.train()


def to_cuda(impl: Any, device: str) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cuda(device)


def to_cpu(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cpu()


def freeze(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = False


def unfreeze(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = True


def get_state_dict(impl: Any) -> Dict[str, Any]:
    rets = {}
    for key in _get_attributes(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            rets[key] = obj.state_dict()
    return rets


def set_state_dict(impl: Any, chkpt: Dict[str, Any]) -> None:
    for key in _get_attributes(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            obj.load_state_dict(chkpt[key])


def reset_optimizer_states(impl: Any) -> None:
    for key in _get_attributes(impl):
        obj = getattr(impl, key)
        if isinstance(obj, torch.optim.Optimizer):
            obj.state = collections.defaultdict(dict)


def map_location(device: str) -> Any:
    if "cuda" in device:
        return lambda storage, loc: storage.cuda(device)
    if "cpu" in device:
        return "cpu"
    raise ValueError(f"invalid device={device}")


class _WithDeviceAndScalerProtocol(Protocol):
    @property
    def device(self) -> str:
        ...

    @property
    def scaler(self) -> Optional[Scaler]:
        ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...


def _convert_to_torch(array: np.ndarray, device: str) -> torch.Tensor:
    dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=array, dtype=dtype, device=device)
    return tensor.float()


class TorchMiniBatch:

    _observations: torch.Tensor
    _actions: torch.Tensor
    _rewards: torch.Tensor
    _next_observations: torch.Tensor
    _terminals: torch.Tensor
    _n_steps: torch.Tensor
    _device: str

    def __init__(
        self,
        batch: TransitionMiniBatch,
        device: str,
        scaler: Optional[Scaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ):
        # convert numpy array to torch tensor
        observations = _convert_to_torch(batch.observations, device)
        actions = _convert_to_torch(batch.actions, device)
        rewards = _convert_to_torch(batch.rewards, device)
        next_observations = _convert_to_torch(batch.next_observations, device)
        terminals = _convert_to_torch(batch.terminals, device)
        n_steps = _convert_to_torch(batch.n_steps, device)

        # apply scaler
        if scaler:
            observations = scaler.transform(observations)
            next_observations = scaler.transform(next_observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._next_observations = next_observations
        self._terminals = terminals
        self._n_steps = n_steps
        self._device = device

    @property
    def observations(self) -> torch.Tensor:
        return self._observations

    @property
    def actions(self) -> torch.Tensor:
        return self._actions

    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards

    @property
    def next_observations(self) -> torch.Tensor:
        return self._next_observations

    @property
    def terminals(self) -> torch.Tensor:
        return self._terminals

    @property
    def n_steps(self) -> torch.Tensor:
        return self._n_steps

    @property
    def device(self) -> str:
        return self._device


def torch_api(
    scaler_targets: Optional[List[str]] = None,
    action_scaler_targets: Optional[List[str]] = None,
    reward_scaler_targets: Optional[List[str]] = None,
) -> Callable[..., np.ndarray]:
    def _torch_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        # get argument names
        sig = signature(f)
        arg_keys = list(sig.parameters.keys())[1:]

        def wrapper(
            self: _WithDeviceAndScalerProtocol, *args: Any, **kwargs: Any
        ) -> np.ndarray:
            tensors: List[Union[torch.Tensor, TorchMiniBatch]] = []

            # convert all args to torch.Tensor
            for i, val in enumerate(args):
                tensor: Union[torch.Tensor, TorchMiniBatch]
                if isinstance(val, torch.Tensor):
                    tensor = val
                elif isinstance(val, list):
                    tensor = default_collate(val)
                    tensor = tensor.to(self.device)
                elif isinstance(val, np.ndarray):
                    if val.dtype == np.uint8:
                        dtype = torch.uint8
                    else:
                        dtype = torch.float32
                    tensor = torch.tensor(
                        data=val,
                        dtype=dtype,
                        device=self.device,
                    )
                elif val is None:
                    tensor = None
                elif isinstance(val, TransitionMiniBatch):
                    tensor = TorchMiniBatch(
                        val,
                        self.device,
                        scaler=self.scaler,
                        action_scaler=self.action_scaler,
                        reward_scaler=self.reward_scaler,
                    )
                else:
                    tensor = torch.tensor(
                        data=val,
                        dtype=torch.float32,
                        device=self.device,
                    )

                if isinstance(tensor, torch.Tensor):
                    # preprocess
                    if self.scaler and scaler_targets:
                        if arg_keys[i] in scaler_targets:
                            tensor = self.scaler.transform(tensor)

                    # preprocess action
                    if self.action_scaler and action_scaler_targets:
                        if arg_keys[i] in action_scaler_targets:
                            tensor = self.action_scaler.transform(tensor)

                    # preprocessing reward
                    if self.reward_scaler and reward_scaler_targets:
                        if arg_keys[i] in reward_scaler_targets:
                            tensor = self.reward_scaler.transform(tensor)

                    # make sure if the tensor is float32 type
                    if tensor is not None and tensor.dtype != torch.float32:
                        tensor = tensor.float()

                tensors.append(tensor)
            return f(self, *tensors, **kwargs)

        return wrapper

    return _torch_api


def eval_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        set_eval_mode(self)
        return f(self, *args, **kwargs)

    return wrapper


def train_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        set_train_mode(self)
        return f(self, *args, **kwargs)

    return wrapper


class View(nn.Module):  # type: ignore

    _shape: Sequence[int]

    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self._shape)


class Swish(nn.Module):  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
