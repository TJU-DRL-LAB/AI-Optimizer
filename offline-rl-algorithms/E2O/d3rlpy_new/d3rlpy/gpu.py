from typing import Any, Dict

import GPUtil

from .context import get_parallel_flag
from .decorators import pretty_repr


def get_gpu_count() -> int:
    return len(GPUtil.getGPUs())


@pretty_repr
class Device:
    """GPU Device class.

    This class manages GPU id.
    The purpose of this device class instead of PyTorch device class is
    to assign GPU ids when the algorithm is trained in parallel with
    scikit-learn utilities such as `sklearn.model_selection.cross_validate` or
    `sklearn.model_selection.GridSearchCV`.

    .. code-block:: python

        from d3rlpy.context import parallel
        from d3rlpy.algos.cql import CQL
        from sklearn.model_selection import cross_validate

        cql = CQL(use_gpu=True)

        # automatically assign different GPUs to parallel training process
        with parallel():
            scores = cross_validate(cql, ..., n_jobs=2)

    Args:
        idx: GPU id.

    """

    _idx: int

    def __init__(self, idx: int = 0):
        self._idx = idx

    def get_id(self) -> int:
        """Returns GPU id.

        Returns:
            GPU id.

        """
        return self._idx

    def __deepcopy__(self, memo: Any) -> "Device":
        if get_parallel_flag():
            # this bahavior is only for sklearn.base.clone
            self._idx += 1
            if self._idx >= get_gpu_count():
                self._idx = 0
        obj = self.__class__(self._idx)
        return obj

    def __eq__(self, obj: Any) -> bool:
        if isinstance(obj, Device):
            return self._idx == obj.get_id()
        raise ValueError("Device cannot be comapred with non Device objects.")

    def __ne__(self, obj: Any) -> bool:
        return not self.__eq__(obj)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"idx": self._idx}
