import copy
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union

from torch import nn

from ..decorators import pretty_repr
from ..torch_utility import Swish
from .torch import (
    Encoder,
    EncoderWithAction,
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)


def _create_activation(activation_type: str) -> nn.Module:
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "swish":
        return Swish()
    raise ValueError("invalid activation_type.")


@pretty_repr
class EncoderFactory:
    TYPE: ClassVar[str] = "none"

    def create(self, observation_shape: Sequence[int]) -> Encoder:
        """Returns PyTorch's state enocder module.

        Args:
            observation_shape: observation shape.

        Returns:
            an enocder object.

        """
        raise NotImplementedError

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        """Returns PyTorch's state-action enocder module.

        Args:
            observation_shape: observation shape.
            action_size: action size. If None, the encoder does not take
                action as input.
            discrete_action: flag if action-space is discrete.

        Returns:
            an enocder object.

        """
        raise NotImplementedError

    def get_type(self) -> str:
        """Returns encoder type.

        Returns:
            encoder type.

        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns encoder parameters.

        Args:
            deep: flag to deeply copy the parameters.

        Returns:
            encoder parameters.

        """
        raise NotImplementedError


class PixelEncoderFactory(EncoderFactory):
    """Pixel encoder factory class.

    This is the default encoder factory for image observation.

    Args:
        filters (list): list of tuples consisting with
            ``(filter_size, kernel_size, stride)``. If None,
            ``Nature DQN``-based architecture is used.
        feature_size (int): the last linear layer size.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "pixel"
    _filters: List[Sequence[int]]
    _feature_size: int
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]

    def __init__(
        self,
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        if filters is None:
            self._filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        else:
            self._filters = filters
        self._feature_size = feature_size
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate

    def create(self, observation_shape: Sequence[int]) -> PixelEncoder:
        assert len(observation_shape) == 3
        return PixelEncoder(
            observation_shape=observation_shape,
            filters=self._filters,
            feature_size=self._feature_size,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            activation=_create_activation(self._activation),
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> PixelEncoderWithAction:
        assert len(observation_shape) == 3
        return PixelEncoderWithAction(
            observation_shape=observation_shape,
            action_size=action_size,
            filters=self._filters,
            feature_size=self._feature_size,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            discrete_action=discrete_action,
            activation=_create_activation(self._activation),
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if deep:
            filters = copy.deepcopy(self._filters)
        else:
            filters = self._filters
        params = {
            "filters": filters,
            "feature_size": self._feature_size,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
        }
        return params


class VectorEncoderFactory(EncoderFactory):
    """Vector encoder factory class.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): list of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        use_dense (bool): flag to use DenseNet architecture.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "vector"
    _hidden_units: Sequence[int]
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool

    def __init__(
        self,
        hidden_units: Optional[Sequence[int]] = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
    ):
        if hidden_units is None:
            self._hidden_units = [256, 256]
        else:
            self._hidden_units = hidden_units
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._use_dense = use_dense

    def create(self, observation_shape: Sequence[int]) -> VectorEncoder:
        assert len(observation_shape) == 1
        return VectorEncoder(
            observation_shape=observation_shape,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            activation=_create_activation(self._activation),
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        assert len(observation_shape) == 1
        return VectorEncoderWithAction(
            observation_shape=observation_shape,
            action_size=action_size,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            discrete_action=discrete_action,
            activation=_create_activation(self._activation),
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if deep:
            hidden_units = copy.deepcopy(self._hidden_units)
        else:
            hidden_units = self._hidden_units
        params = {
            "hidden_units": hidden_units,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
            "use_dense": self._use_dense,
        }
        return params


class DefaultEncoderFactory(EncoderFactory):
    """Default encoder factory class.

    This encoder factory returns an encoder based on observation shape.

    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "default"
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]

    def __init__(
        self,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate

    def create(self, observation_shape: Sequence[int]) -> Encoder:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        return factory.create(observation_shape)

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self._activation,
                use_batch_norm=self._use_batch_norm,
                dropout_rate=self._dropout_rate,
            )
        return factory.create_with_action(
            observation_shape, action_size, discrete_action
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
        }


class DenseEncoderFactory(EncoderFactory):
    """DenseNet encoder factory class.

    This is an alias for DenseNet architecture proposed in D2RL.
    This class does exactly same as follows.

    .. code-block:: python

       from d3rlpy.encoders import VectorEncoderFactory

       factory = VectorEncoderFactory(hidden_units=[256, 256, 256, 256],
                                      use_dense=True)

    For now, this only supports vector observations.

    References:
        * `Sinha et al., D2RL: Deep Dense Architectures in Reinforcement
          Learning. <https://arxiv.org/abs/2010.09163>`_

    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.

    """

    TYPE: ClassVar[str] = "dense"
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]

    def __init__(
        self,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate

    def create(self, observation_shape: Sequence[int]) -> VectorEncoder:
        if len(observation_shape) == 3:
            raise NotImplementedError("pixel observation is not supported.")
        factory = VectorEncoderFactory(
            hidden_units=[256, 256, 256, 256],
            activation=self._activation,
            use_dense=True,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
        )
        return factory.create(observation_shape)

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        if len(observation_shape) == 3:
            raise NotImplementedError("pixel observation is not supported.")
        factory = VectorEncoderFactory(
            hidden_units=[256, 256, 256, 256],
            activation=self._activation,
            use_dense=True,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
        )
        return factory.create_with_action(
            observation_shape, action_size, discrete_action
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
        }


ENCODER_LIST: Dict[str, Type[EncoderFactory]] = {}


def register_encoder_factory(cls: Type[EncoderFactory]) -> None:
    """Registers encoder factory class.

    Args:
        cls: encoder factory class inheriting ``EncoderFactory``.

    """
    is_registered = cls.TYPE in ENCODER_LIST
    assert not is_registered, f"{cls.TYPE} seems to be already registered"
    ENCODER_LIST[cls.TYPE] = cls


def create_encoder_factory(name: str, **kwargs: Any) -> EncoderFactory:
    """Returns registered encoder factory object.

    Args:
        name: regsitered encoder factory type name.
        kwargs: encoder arguments.

    Returns:
        encoder factory object.

    """
    assert name in ENCODER_LIST, f"{name} seems not to be registered."
    factory = ENCODER_LIST[name](**kwargs)
    assert isinstance(factory, EncoderFactory)
    return factory


register_encoder_factory(VectorEncoderFactory)
register_encoder_factory(PixelEncoderFactory)
register_encoder_factory(DefaultEncoderFactory)
register_encoder_factory(DenseEncoderFactory)
