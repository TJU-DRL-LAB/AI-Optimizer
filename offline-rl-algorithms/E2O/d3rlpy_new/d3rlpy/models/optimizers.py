import copy
from typing import Any, Dict, Iterable, Tuple, Type, Union, cast

from torch import nn, optim
from torch.optim import SGD, Adam, Optimizer, RMSprop

from ..decorators import pretty_repr


@pretty_repr
class OptimizerFactory:
    """A factory class that creates an optimizer object in a lazy way.

    The optimizers in algorithms can be configured through this factory class.

    .. code-block:: python

        from torch.optim Adam
        from d3rlpy.optimizers import OptimizerFactory
        from d3rlpy.algos import DQN

        factory = OptimizerFactory(Adam, eps=0.001)

        dqn = DQN(optim_factory=factory)

    Args:
        optim_cls: An optimizer class.
        kwargs: arbitrary keyword-arguments.

    """

    _optim_cls: Type[Optimizer]
    _optim_kwargs: Dict[str, Any]

    def __init__(self, optim_cls: Union[Type[Optimizer], str], **kwargs: Any):
        if isinstance(optim_cls, str):
            self._optim_cls = cast(Type[Optimizer], getattr(optim, optim_cls))
        else:
            self._optim_cls = optim_cls
        self._optim_kwargs = kwargs

    def create(self, params: Iterable[nn.Parameter], lr: float) -> Optimizer:
        """Returns an optimizer object.

        Args:
            params (list): a list of PyTorch parameters.
            lr (float): learning rate.

        Returns:
            torch.optim.Optimizer: an optimizer object.

        """
        return self._optim_cls(params, lr=lr, **self._optim_kwargs)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns optimizer parameters.

        Args:
            deep: flag to deeply copy the parameters.

        Returns:
            optimizer parameters.

        """
        if deep:
            params = copy.deepcopy(self._optim_kwargs)
        else:
            params = self._optim_kwargs
        return {"optim_cls": self._optim_cls.__name__, **params}


class SGDFactory(OptimizerFactory):
    """An alias for SGD optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import SGDFactory

        factory = SGDFactory(weight_decay=1e-4)

    Args:
        momentum: momentum factor.
        dampening: dampening for momentum.
        weight_decay: weight decay (L2 penalty).
        nesterov: flag to enable Nesterov momentum.

    """

    def __init__(
        self,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        **kwargs: Any
    ):
        super().__init__(
            optim_cls=SGD,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


class AdamFactory(OptimizerFactory):
    """An alias for Adam optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamFactory

        factory = AdamFactory(weight_decay=1e-4)

    Args:
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.

    """

    def __init__(
        self,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        **kwargs: Any
    ):
        super().__init__(
            optim_cls=Adam,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


class RMSpropFactory(OptimizerFactory):
    """An alias for RMSprop optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import RMSpropFactory

        factory = RMSpropFactory(weight_decay=1e-4)

    Args:
        alpha: smoothing constant.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        momentum: momentum factor.
        centered: flag to compute the centered RMSProp, the gradient is
            normalized by an estimation of its variance.

    """

    def __init__(
        self,
        alpha: float = 0.95,
        eps: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = True,
        **kwargs: Any
    ):
        super().__init__(
            optim_cls=RMSprop,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
