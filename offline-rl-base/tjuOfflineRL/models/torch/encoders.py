from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...itertools import last_flag
from ...torch_utility import View


class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        raise NotImplementedError

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError


class EncoderWithAction(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        pass

    @property
    def action_size(self) -> int:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        raise NotImplementedError

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError


class _PixelEncoder(nn.Module):  # type: ignore

    _observation_shape: Sequence[int]
    _feature_size: int
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _activation: nn.Module
    _convs: nn.ModuleList
    _conv_bns: nn.ModuleList
    _fc: nn.Linear
    _fc_bn: nn.BatchNorm1d
    _dropouts: nn.ModuleList

    def __init__(
        self,
        observation_shape: Sequence[int],
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        # default architecture is based on Nature DQN paper.
        if filters is None:
            filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        if feature_size is None:
            feature_size = 512

        self._observation_shape = observation_shape
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._activation = activation
        self._feature_size = feature_size

        # convolutional layers
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        self._convs = nn.ModuleList()
        self._conv_bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size, stride=stride
            )
            self._convs.append(conv)

            # use batch normalization layer
            if use_batch_norm:
                self._conv_bns.append(nn.BatchNorm2d(out_channel))

            # use dropout layer
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout2d(dropout_rate))

        # last dense layer
        self._fc = nn.Linear(self._get_linear_input_size(), feature_size)
        if use_batch_norm:
            self._fc_bn = nn.BatchNorm1d(feature_size)
        if dropout_rate is not None:
            self._dropouts.append(nn.Dropout(dropout_rate))

    def _get_linear_input_size(self) -> int:
        x = torch.rand((1,) + tuple(self._observation_shape))
        with torch.no_grad():
            return self._conv_encode(x).view(1, -1).shape[1]  # type: ignore

    def _get_last_conv_shape(self) -> Sequence[int]:
        x = torch.rand((1,) + tuple(self._observation_shape))
        with torch.no_grad():
            return self._conv_encode(x).shape  # type: ignore

    def _conv_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self._convs):
            h = self._activation(conv(h))
            if self._use_batch_norm:
                h = self._conv_bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def last_layer(self) -> nn.Linear:
        return self._fc


class PixelEncoder(_PixelEncoder, Encoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._conv_encode(x)

        h = self._activation(self._fc(h.view(h.shape[0], -1)))
        if self._use_batch_norm:
            h = self._fc_bn(h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)

        return h

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        modules: List[torch.nn.Module] = []

        # add linear layer
        modules.append(nn.Linear(self.get_feature_size(), self._fc.in_features))
        modules.append(self._activation)

        # reshape output
        modules.append(View((-1, *self._get_last_conv_shape()[1:])))

        # add conv layers
        for is_last, conv in last_flag(reversed(self._convs)):
            deconv = nn.ConvTranspose2d(
                conv.out_channels,
                conv.in_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
            )
            modules.append(deconv)

            if not is_last:
                modules.append(self._activation)

        return modules


class PixelEncoderWithAction(_PixelEncoder, EncoderWithAction):

    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        super().__init__(
            observation_shape=observation_shape,
            filters=filters,
            feature_size=feature_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation=activation,
        )

    def _get_linear_input_size(self) -> int:
        size = super()._get_linear_input_size()
        return size + self._action_size

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._conv_encode(x)

        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()

        # cocat feature and action
        h = torch.cat([h.view(h.shape[0], -1), action], dim=1)
        h = self._activation(self._fc(h))
        if self._use_batch_norm:
            h = self._fc_bn(h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)

        return h

    @property
    def action_size(self) -> int:
        return self._action_size

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        modules: List[torch.nn.Module] = []

        # add linear layer
        in_features = self._fc.in_features - self._action_size
        modules.append(nn.Linear(self.get_feature_size(), in_features))
        modules.append(self._activation)

        # reshape output
        modules.append(View((-1, *self._get_last_conv_shape()[1:])))

        # add conv layers
        for is_last, conv in last_flag(reversed(self._convs)):
            deconv = nn.ConvTranspose2d(
                conv.out_channels,
                conv.in_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
            )
            modules.append(deconv)

            if not is_last:
                modules.append(self._activation)

        return modules


class _VectorEncoder(nn.Module):  # type: ignore

    _observation_shape: Sequence[int]
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool
    _activation: nn.Module
    _feature_size: int
    _fcs: nn.ModuleList
    _bns: nn.ModuleList
    _dropouts: nn.ModuleList

    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self._observation_shape = observation_shape

        if hidden_units is None:
            hidden_units = [256, 256]

        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._feature_size = hidden_units[-1]
        self._activation = activation
        self._use_dense = use_dense

        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        self._fcs = nn.ModuleList()
        self._bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
            if use_dense and i > 0:
                in_unit += observation_shape[0]
            self._fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout(dropout_rate))

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, fc in enumerate(self._fcs):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def last_layer(self) -> nn.Linear:
        return self._fcs[-1]


class VectorEncoder(_VectorEncoder, Encoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        assert not self._use_dense, "use_dense=True is not supported yet"
        modules: List[torch.nn.Module] = []
        for is_last, fc in last_flag(reversed(self._fcs)):
            modules.append(nn.Linear(fc.out_features, fc.in_features))
            if not is_last:
                modules.append(self._activation)
        return modules


class VectorEncoderWithAction(_VectorEncoder, EncoderWithAction):

    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        concat_shape = (observation_shape[0] + action_size,)
        super().__init__(
            observation_shape=concat_shape,
            hidden_units=hidden_units,
            use_batch_norm=use_batch_norm,
            use_dense=use_dense,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self.action_size
            ).float()
        x = torch.cat([x, action], dim=1)
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    @property
    def action_size(self) -> int:
        return self._action_size

    def create_reverse(self) -> Sequence[torch.nn.Module]:
        assert not self._use_dense, "use_dense=True is not supported yet"
        modules: List[torch.nn.Module] = []
        for is_last, fc in last_flag(reversed(self._fcs)):
            if is_last:
                in_features = fc.in_features - self._action_size
            else:
                in_features = fc.in_features

            modules.append(nn.Linear(fc.out_features, in_features))

            if not is_last:
                modules.append(self._activation)
        return modules
