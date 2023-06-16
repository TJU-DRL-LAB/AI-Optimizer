# pylint: disable=redefined-builtin,exec-used

import glob
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import click
import gym
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from . import algos
from ._version import __version__
from .envs import Monitor
from .metrics.scorer import evaluate_on_environment

if TYPE_CHECKING:
    import matplotlib.pyplot


def print_stats(path: str) -> None:
    data = np.loadtxt(path, delimiter=",")
    print("FILE NAME  : ", path)
    print("EPOCH      : ", data[-1, 0])
    print("TOTAL STEPS: ", data[-1, 1])
    print("MAX VALUE  : ", np.max(data[:, 2]))
    print("MIN VALUE  : ", np.min(data[:, 2]))
    print("STD VALUE  : ", np.std(data[:, 2]))


def get_plt() -> "matplotlib.pyplot":
    import matplotlib.pyplot as plt

    try:
        # enable seaborn style if available
        import seaborn as sns

        sns.set()
    except ImportError:
        pass
    return plt


@click.group()
def cli() -> None:
    print(f"d3rlpy command line interface (Version {__version__})")


@cli.command(short_help="Show statistics of save metrics.")
@click.argument("path")
def stats(path: str) -> None:
    print_stats(path)


@cli.command(short_help="Plot saved metrics (requires matplotlib).")
@click.argument("path", nargs=-1)
@click.option(
    "--window", default=1, show_default=True, help="moving average window."
)
@click.option("--show-steps", is_flag=True, help="use iterations on x-axis.")
@click.option("--show-max", is_flag=True, help="show maximum value.")
@click.option("--label", multiple=True, help="label in legend.")
@click.option("--xlim", nargs=2, type=float, help="limit on x-axis (tuple).")
@click.option("--ylim", nargs=2, type=float, help="limit on y-axis (tuple).")
@click.option("--title", help="title of the plot.")
@click.option("--ylabel", default="value", help="label on y-axis.")
@click.option("--save", help="flag to save the plot as an image.")
def plot(
    path: List[str],
    window: int,
    show_steps: bool,
    show_max: bool,
    label: Optional[Sequence[str]],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    title: Optional[str],
    ylabel: str,
    save: str,
) -> None:
    plt = get_plt()

    max_y_values = []
    min_x_values = []
    max_x_values = []

    if label:
        assert len(label) == len(
            path
        ), "--labels must be provided as many as the number of paths"

    for i, p in enumerate(path):
        data = np.loadtxt(p, delimiter=",")

        # filter to smooth data
        y_data = uniform_filter1d(data[:, 2], size=window)

        # create label
        if label:
            _label = label[i]
        elif len(p.split(os.sep)) > 1:
            _label = "/".join(p.split(os.sep)[-2:])
        else:
            _label = p

        if show_steps:
            x_data = data[:, 1]
        else:
            x_data = data[:, 0]

        max_y_values.append(np.max(data[:, 2]))
        min_x_values.append(np.min(x_data))
        max_x_values.append(np.max(x_data))

        # show statistics
        print("")
        print_stats(p)

        plt.plot(x_data, y_data, label=_label)

    if show_max:
        plt.plot(
            [np.min(min_x_values), np.max(max_x_values)],
            [np.max(max_y_values), np.max(max_y_values)],
            color="black",
            linestyle="dashed",
        )

    plt.xlabel("steps" if show_steps else "epochs")
    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if title:
        plt.title(title)

    plt.legend()
    if save:
        plt.savefig(save)
    else:
        plt.show()


@cli.command(short_help="Plot saved metrics in a grid (requires matplotlib).")
@click.argument("path")
@click.option("--title", help="title of the plot.")
@click.option("--save", help="flag to save the plot as an image.")
def plot_all(
    path: str,
    title: Optional[str],
    save: str,
) -> None:
    plt = get_plt()

    # print params.json
    if os.path.exists(os.path.join(path, "params.json")):
        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.loads(f.read())
        print("")
        for k, v in params.items():
            print(f"{k}={v}")

    metrics_names = sorted(list(glob.glob(os.path.join(path, "*.csv"))))
    n_cols = int(np.ceil(len(metrics_names) ** 0.5))
    n_rows = int(np.ceil(len(metrics_names) / n_cols))

    plt.figure(figsize=(12, 7))

    for i in range(n_rows):
        for j in range(n_cols):
            index = j + n_cols * i
            if index >= len(metrics_names):
                break

            plt.subplot(n_rows, n_cols, index + 1)

            data = np.loadtxt(metrics_names[index], delimiter=",")

            plt.plot(data[:, 0], data[:, 2])
            plt.title(os.path.basename(metrics_names[index]))
            plt.xlabel("epoch")
            plt.ylabel("value")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()


def _get_params_json_path(path: str) -> str:
    dirname = os.path.dirname(path)
    if not os.path.exists(os.path.join(dirname, "params.json")):
        raise RuntimeError(
            "params.json is not found in %s. Please specify"
            "the path to params.json by --params-json."
        )
    return os.path.join(dirname, "params.json")


@cli.command(short_help="Export saved model as inference model format.")
@click.argument("path")
@click.option(
    "--format",
    default="onnx",
    show_default=True,
    help="model format (torchscript, onnx).",
)
@click.option(
    "--params-json", default=None, help="explicitly specify params.json."
)
@click.option("--out", default=None, help="output path.")
def export(
    path: str, format: str, params_json: Optional[str], out: Optional[str]
) -> None:
    # check format
    if format not in ["onnx", "torchscript"]:
        raise ValueError("Please specify onnx or torchscript.")

    # find params.json
    if params_json is None:
        params_json = _get_params_json_path(path)

    # load params
    with open(params_json, "r") as f:
        params = json.loads(f.read())

    # load saved model
    print(f"Loading {path}...")
    algo = getattr(algos, params["algorithm"]).from_json(params_json)
    algo.load_model(path)

    if out is None:
        ext = "onnx" if format == "onnx" else "torchscript"
        export_name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(os.path.dirname(path), export_name + "." + ext)

    # export inference model
    print(f"Exporting to {out}...")
    algo.save_policy(out, as_onnx=format == "onnx")


def _exec_to_create_env(code: str) -> gym.Env:
    print(f"Executing '{code}'")
    variables: Dict[str, Any] = {}
    exec(code, globals(), variables)
    if "env" not in variables:
        raise RuntimeError("env must be defined in env_header.")
    return variables["env"]


@cli.command(short_help="Record episodes with the saved model.")
@click.argument("model_path")
@click.option("--env-id", default=None, help="Gym environment id.")
@click.option(
    "--env-header", default=None, help="one-liner to create environment."
)
@click.option("--out", default="videos", help="output directory path.")
@click.option(
    "--params-json", default=None, help="explicityly specify params.json."
)
@click.option(
    "--n-episodes", default=3, help="the number of episodes to record."
)
@click.option("--frame-rate", default=60, help="video frame rate.")
@click.option("--record-rate", default=1, help="record frame rate.")
@click.option("--epsilon", default=0.0, help="epsilon-greedy evaluation.")
def record(
    model_path: str,
    env_id: Optional[str],
    env_header: Optional[str],
    params_json: Optional[str],
    out: str,
    n_episodes: int,
    frame_rate: float,
    record_rate: int,
    epsilon: float,
) -> None:
    if params_json is None:
        params_json = _get_params_json_path(model_path)

    # load params
    with open(params_json, "r") as f:
        params = json.loads(f.read())

    # load saved model
    print(f"Loading {model_path}...")
    algo = getattr(algos, params["algorithm"]).from_json(params_json)
    algo.load_model(model_path)

    # wrap environment with Monitor
    env: gym.Env
    if env_id is not None:
        env = gym.make(env_id)
    elif env_header is not None:
        env = _exec_to_create_env(env_header)
    else:
        raise ValueError("env_id or env_header must be provided.")

    wrapped_env = Monitor(
        env,
        out,
        video_callable=lambda ep: ep % 1 == 0,
        frame_rate=float(frame_rate),
        record_rate=int(record_rate),
    )

    # run episodes
    evaluate_on_environment(wrapped_env, n_episodes, epsilon=epsilon)(algo)


@cli.command(short_help="Run evaluation episodes with rendering.")
@click.argument("model_path")
@click.option("--env-id", default=None, help="Gym environment id.")
@click.option(
    "--env-header", default=None, help="one-liner to create environment."
)
@click.option(
    "--params-json", default=None, help="explicityly specify params.json."
)
@click.option("--n-episodes", default=3, help="the number of episodes to run.")
def play(
    model_path: str,
    env_id: Optional[str],
    env_header: Optional[str],
    params_json: Optional[str],
    n_episodes: int,
) -> None:
    if params_json is None:
        params_json = _get_params_json_path(model_path)

    # load params
    with open(params_json, "r") as f:
        params = json.loads(f.read())

    # load saved model
    print(f"Loading {model_path}...")
    algo = getattr(algos, params["algorithm"]).from_json(params_json)
    algo.load_model(model_path)

    # wrap environment with Monitor
    env: gym.Env
    if env_id is not None:
        env = gym.make(env_id)
    elif env_header is not None:
        env = _exec_to_create_env(env_header)
    else:
        raise ValueError("env_id or env_header must be provided.")

    # run episodes
    evaluate_on_environment(env, n_episodes, render=True)(algo)
