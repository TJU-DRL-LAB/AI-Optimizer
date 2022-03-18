from typing import Type, TypeVar

T = TypeVar("T")


def pretty_repr(cls: Type[T]) -> Type[T]:
    assert hasattr(cls, "get_params"), f"{cls} must have get_params method."

    def _repr(self: T) -> str:
        pairs = []
        for k, v in self.get_params(deep=False).items():  # type: ignore
            if isinstance(v, str):
                pairs.append(f"{k}='{v}'")
            else:
                pairs.append(f"{k}={v}")
        params_str = ", ".join(pairs)
        module_name = self.__class__.__module__
        cls_name = self.__class__.__name__
        return f"{module_name}.{cls_name}({params_str})"

    cls.__repr__ = _repr  # type: ignore

    return cls
