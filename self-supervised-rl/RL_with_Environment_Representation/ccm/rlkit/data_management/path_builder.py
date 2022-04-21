import numpy as np


class PathBuilder(dict):
    """
    Usage:
    ```
    path_builder = PathBuilder()
    path.add_sample(
        observations=1,
        actions=2,
        next_observations=3,
        ...
    )
    path.add_sample(
        observations=4,
        actions=5,
        next_observations=6,
        ...
    )

    path = path_builder.get_all_stacked()

    path['observations']
    # output: [1, 4]
    path['actions']
    # output: [2, 5]
    ```

    Note that the key should be "actions" and not "action" since the
    resulting dictionary will have those keys.
    """

    def __init__(self):
        super().__init__()
        self._path_length = 0

    def add_all(self, **key_to_value):
        for k, v in key_to_value.items():
            if k not in self:
                self[k] = [v]
            else:
                self[k].append(v)
        self._path_length += 1

    def get_all_stacked(self):
        output_dict = dict()
        for k, v in self.items():
            output_dict[k] = stack_list(v)
        return output_dict

    def __len__(self):
        return self._path_length


def stack_list(lst):
    if isinstance(lst[0], dict):
        return lst
    else:
        return np.array(lst)
