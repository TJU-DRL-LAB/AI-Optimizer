from typing import Any, ClassVar, Dict, Type

from ..decorators import pretty_repr
from .torch import (
    ContinuousFQFQFunction,
    ContinuousIQNQFunction,
    ContinuousMeanQFunction,
    ContinuousQFunction,
    ContinuousQRQFunction,
    DiscreteFQFQFunction,
    DiscreteIQNQFunction,
    DiscreteMeanQFunction,
    DiscreteQFunction,
    DiscreteQRQFunction,
    Encoder,
    EncoderWithAction,
)


@pretty_repr
class QFunctionFactory:
    TYPE: ClassVar[str] = "none"

    _share_encoder: bool

    def __init__(self, share_encoder: bool):
        self._share_encoder = share_encoder

    def create_discrete(
        self, encoder: Encoder, action_size: int
    ) -> DiscreteQFunction:
        """Returns PyTorch's Q function module.

        Args:
            encoder: an encoder module that processes the observation to
                obtain feature representations.
            action_size: dimension of discrete action-space.

        Returns:
            discrete Q function object.

        """
        raise NotImplementedError

    def create_continuous(
        self, encoder: EncoderWithAction
    ) -> ContinuousQFunction:
        """Returns PyTorch's Q function module.

        Args:
            encoder: an encoder module that processes the observation and
                action to obtain feature representations.

        Returns:
            continuous Q function object.

        """
        raise NotImplementedError

    def get_type(self) -> str:
        """Returns Q function type.

        Returns:
            Q function type.

        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns Q function parameters.

        Returns:
            Q function parameters.

        """
        raise NotImplementedError

    @property
    def share_encoder(self) -> bool:
        return self._share_encoder


class MeanQFunctionFactory(QFunctionFactory):
    """Standard Q function factory class.

    This is the standard Q function factory class.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.

    """

    TYPE: ClassVar[str] = "mean"

    def __init__(self, share_encoder: bool = False, **kwargs: Any):
        super().__init__(share_encoder)

    def create_discrete(
        self,
        encoder: Encoder,
        action_size: int,
    ) -> DiscreteMeanQFunction:
        return DiscreteMeanQFunction(encoder, action_size)

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousMeanQFunction:
        return ContinuousMeanQFunction(encoder)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "share_encoder": self._share_encoder,
        }


class QRQFunctionFactory(QFunctionFactory):
    """Quantile Regression Q function factory class.

    References:
        * `Dabney et al., Distributional reinforcement learning with quantile
          regression. <https://arxiv.org/abs/1710.10044>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.

    """

    TYPE: ClassVar[str] = "qr"
    _n_quantiles: int

    def __init__(
        self, share_encoder: bool = False, n_quantiles: int = 32, **kwargs: Any
    ):
        super().__init__(share_encoder)
        self._n_quantiles = n_quantiles

    def create_discrete(
        self, encoder: Encoder, action_size: int
    ) -> DiscreteQRQFunction:
        return DiscreteQRQFunction(encoder, action_size, self._n_quantiles)

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousQRQFunction:
        return ContinuousQRQFunction(encoder, self._n_quantiles)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "share_encoder": self._share_encoder,
            "n_quantiles": self._n_quantiles,
        }

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles


class IQNQFunctionFactory(QFunctionFactory):
    """Implicit Quantile Network Q function factory class.

    References:
        * `Dabney et al., Implicit quantile networks for distributional
          reinforcement learning. <https://arxiv.org/abs/1806.06923>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
        n_greedy_quantiles: the number of quantiles for inference.
        embed_size: the embedding size.

    """

    TYPE: ClassVar[str] = "iqn"
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int

    def __init__(
        self,
        share_encoder: bool = False,
        n_quantiles: int = 64,
        n_greedy_quantiles: int = 32,
        embed_size: int = 64,
        **kwargs: Any,
    ):
        super().__init__(share_encoder)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size

    def create_discrete(
        self,
        encoder: Encoder,
        action_size: int,
    ) -> DiscreteIQNQFunction:
        return DiscreteIQNQFunction(
            encoder=encoder,
            action_size=action_size,
            n_quantiles=self._n_quantiles,
            n_greedy_quantiles=self._n_greedy_quantiles,
            embed_size=self._embed_size,
        )

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousIQNQFunction:
        return ContinuousIQNQFunction(
            encoder=encoder,
            n_quantiles=self._n_quantiles,
            n_greedy_quantiles=self._n_greedy_quantiles,
            embed_size=self._embed_size,
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "share_encoder": self._share_encoder,
            "n_quantiles": self._n_quantiles,
            "n_greedy_quantiles": self._n_greedy_quantiles,
            "embed_size": self._embed_size,
        }

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    @property
    def n_greedy_quantiles(self) -> int:
        return self._n_greedy_quantiles

    @property
    def embed_size(self) -> int:
        return self._embed_size


class FQFQFunctionFactory(QFunctionFactory):
    """Fully parameterized Quantile Function Q function factory.

    References:
        * `Yang et al., Fully parameterized quantile function for
          distributional reinforcement learning.
          <https://arxiv.org/abs/1911.02140>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
        embed_size: the embedding size.
        entropy_coeff: the coefficiency of entropy penalty term.

    """

    TYPE: ClassVar[str] = "fqf"
    _n_quantiles: int
    _embed_size: int
    _entropy_coeff: float

    def __init__(
        self,
        share_encoder: bool = False,
        n_quantiles: int = 32,
        embed_size: int = 64,
        entropy_coeff: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(share_encoder)
        self._n_quantiles = n_quantiles
        self._embed_size = embed_size
        self._entropy_coeff = entropy_coeff

    def create_discrete(
        self,
        encoder: Encoder,
        action_size: int,
    ) -> DiscreteFQFQFunction:
        return DiscreteFQFQFunction(
            encoder=encoder,
            action_size=action_size,
            n_quantiles=self._n_quantiles,
            embed_size=self._embed_size,
            entropy_coeff=self._entropy_coeff,
        )

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousFQFQFunction:
        return ContinuousFQFQFunction(
            encoder=encoder,
            n_quantiles=self._n_quantiles,
            embed_size=self._embed_size,
            entropy_coeff=self._entropy_coeff,
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "share_encoder": self._share_encoder,
            "n_quantiles": self._n_quantiles,
            "embed_size": self._embed_size,
            "entropy_coeff": self._entropy_coeff,
        }

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    @property
    def embed_size(self) -> int:
        return self._embed_size

    @property
    def entropy_coeff(self) -> float:
        return self._entropy_coeff


Q_FUNC_LIST: Dict[str, Type[QFunctionFactory]] = {}


def register_q_func_factory(cls: Type[QFunctionFactory]) -> None:
    """Registers Q function factory class.

    Args:
        cls: Q function factory class inheriting ``QFunctionFactory``.

    """
    is_registered = cls.TYPE in Q_FUNC_LIST
    assert not is_registered, f"{cls.TYPE} seems to be already registered"
    Q_FUNC_LIST[cls.TYPE] = cls


def create_q_func_factory(name: str, **kwargs: Any) -> QFunctionFactory:
    """Returns registered Q function factory object.

    Args:
        name: registered Q function factory type name.
        kwargs: Q function arguments.

    Returns:
        Q function factory object.

    """
    assert name in Q_FUNC_LIST, f"{name} seems not to be registered."
    factory = Q_FUNC_LIST[name](**kwargs)
    assert isinstance(factory, QFunctionFactory)
    return factory


register_q_func_factory(MeanQFunctionFactory)
register_q_func_factory(QRQFunctionFactory)
register_q_func_factory(IQNQFunctionFactory)
register_q_func_factory(FQFQFunctionFactory)
