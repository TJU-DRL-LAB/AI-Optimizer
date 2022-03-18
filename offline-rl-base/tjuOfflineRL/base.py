import copy
import json
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import gym
import numpy as np
from tqdm.auto import tqdm

from .argument_utility import (
    ActionScalerArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_action_scaler,
    check_reward_scaler,
    check_scaler,
)
from .constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from .context import disable_parallel
from .dataset import Episode, MDPDataset, Transition, TransitionMiniBatch
from .decorators import pretty_repr
from .gpu import Device
from .iterators import RandomIterator, RoundIterator, TransitionIterator
from .logger import LOG, TJUOfflineRLLogger
from .models.encoders import EncoderFactory, create_encoder_factory
from .models.optimizers import OptimizerFactory
from .models.q_functions import QFunctionFactory, create_q_func_factory
from .online.utility import get_action_size_from_env
from .preprocessing import (
    ActionScaler,
    RewardScaler,
    Scaler,
    create_action_scaler,
    create_reward_scaler,
    create_scaler,
)


class ImplBase(metaclass=ABCMeta):
    @abstractmethod
    def save_model(self, fname: str) -> None:
        pass

    @abstractmethod
    def load_model(self, fname: str) -> None:
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> Sequence[int]:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in params.items():
        if isinstance(value, Device):
            params[key] = value.get_id()
        elif isinstance(
            value,
            (
                Scaler,
                ActionScaler,
                RewardScaler,
                EncoderFactory,
                QFunctionFactory,
            ),
        ):
            params[key] = {
                "type": value.get_type(),
                "params": value.get_params(),
            }
        elif isinstance(value, OptimizerFactory):
            params[key] = value.get_params()
    return params


def _deseriealize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in params.items():
        if key == "scaler" and params["scaler"]:
            scaler_type = params["scaler"]["type"]
            scaler_params = params["scaler"]["params"]
            scaler = create_scaler(scaler_type, **scaler_params)
            params[key] = scaler
        elif key == "action_scaler" and params["action_scaler"]:
            scaler_type = params["action_scaler"]["type"]
            scaler_params = params["action_scaler"]["params"]
            action_scaler = create_action_scaler(scaler_type, **scaler_params)
            params[key] = action_scaler
        elif key == "reward_scaler" and params["reward_scaler"]:
            scaler_type = params["reward_scaler"]["type"]
            scaler_params = params["reward_scaler"]["params"]
            reward_scaler = create_reward_scaler(scaler_type, **scaler_params)
            params[key] = reward_scaler
        elif "optim_factory" in key:
            params[key] = OptimizerFactory(**value)
        elif "encoder_factory" in key:
            params[key] = create_encoder_factory(
                value["type"], **value["params"]
            )
        elif key == "q_func_factory":
            params[key] = create_q_func_factory(
                value["type"], **value["params"]
            )
    return params


@pretty_repr
class LearnableBase:

    _batch_size: int
    _n_frames: int
    _n_steps: int
    _gamma: float
    _scaler: Optional[Scaler]
    _action_scaler: Optional[ActionScaler]
    _reward_scaler: Optional[RewardScaler]
    _real_ratio: float
    _generated_maxlen: int
    _impl: Optional[ImplBase]
    _eval_results: DefaultDict[str, List[float]]
    _loss_history: DefaultDict[str, List[float]]
    _active_logger: Optional[TJUOfflineRLLogger]
    _grad_step: int

    def __init__(
        self,
        batch_size: int,
        n_frames: int,
        n_steps: int,
        gamma: float,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._batch_size = batch_size
        self._n_frames = n_frames
        self._n_steps = n_steps
        self._gamma = gamma
        self._scaler = check_scaler(scaler)
        self._action_scaler = check_action_scaler(action_scaler)
        self._reward_scaler = check_reward_scaler(reward_scaler)
        self._real_ratio = real_ratio
        self._generated_maxlen = generated_maxlen

        self._impl = None
        self._eval_results = defaultdict(list)
        self._loss_history = defaultdict(list)
        self._active_logger = None
        self._grad_step = 0

        if kwargs and len(kwargs.keys()) > 0:
            LOG.warning("Unused arguments are passed.", **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # propagate property updates to implementation object
        if hasattr(self, "_impl") and self._impl and hasattr(self._impl, name):
            setattr(self._impl, name, value)

    @classmethod
    def from_json(
        cls, fname: str, use_gpu: UseGPUArg = False
    ) -> "LearnableBase":
        """Returns algorithm configured with json file.

        The Json file should be the one saved during fitting.

        .. code-block:: python

            from tjuOfflineRL.algos import Algo

            # create algorithm with saved configuration
            algo = Algo.from_json('tjuOfflineRL_logs/<path-to-json>/params.json')

            # ready to load
            algo.load_model('tjuOfflineRL_logs/<path-to-model>/model_100.pt')

            # ready to predict
            algo.predict(...)

        Args:
            fname: file path to `params.json`.
            use_gpu: flag to use GPU, device ID or device.

        Returns:
            algorithm.

        """
        with open(fname, "r") as f:
            params = json.load(f)

        observation_shape = tuple(params["observation_shape"])
        action_size = params["action_size"]
        del params["observation_shape"]
        del params["action_size"]

        # reconstruct objects from json
        params = _deseriealize_params(params)

        # overwrite use_gpu flag
        params["use_gpu"] = use_gpu

        algo = cls(**params)
        algo.create_impl(observation_shape, action_size)
        return algo

    def set_params(self, **params: Any) -> "LearnableBase":
        """Sets the given arguments to the attributes if they exist.

        This method sets the given values to the attributes including ones in
        subclasses. If the values that don't exist as attributes are
        passed, they are ignored.
        Some of scikit-learn utilities will use this method.

        .. code-block:: python

            algo.set_params(batch_size=100)

        Args:
            params: arbitrary inputs to set as attributes.

        Returns:
            itself.

        """
        for key, val in params.items():
            if hasattr(self, key):
                try:
                    setattr(self, key, val)
                except AttributeError:
                    # try passing to protected keys
                    assert hasattr(self, "_" + key), f"{key} does not exist."
                    setattr(self, "_" + key, val)
            else:
                assert hasattr(self, "_" + key), f"{key} does not exist."
                setattr(self, "_" + key, val)
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Returns the all attributes.

        This method returns the all attributes including ones in subclasses.
        Some of scikit-learn utilities will use this method.

        .. code-block:: python

            params = algo.get_params(deep=True)

            # the returned values can be used to instantiate the new object.
            algo2 = AlgoBase(**params)

        Args:
            deep: flag to deeply copy objects such as `impl`.

        Returns:
            attribute values in dictionary.

        """
        rets = {}
        for key in dir(self):
            # remove magic properties
            if key[:2] == "__":
                continue

            # remove specific keys
            if key in [
                "_eval_results",
                "_loss_history",
                "_active_logger",
                "_grad_step",
                "active_logger",
                "grad_step",
                "observation_shape",
                "action_size",
            ]:
                continue

            value = getattr(self, key)

            # remove underscore
            if key[0] == "_":
                key = key[1:]

            # pick scalar parameters
            if np.isscalar(value):
                rets[key] = value
            elif isinstance(value, object) and not callable(value):
                if deep:
                    rets[key] = copy.deepcopy(value)
                else:
                    rets[key] = value
        return rets

    def save_model(self, fname: str) -> None:
        """Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname: destination file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.save_model(fname)

    def load_model(self, fname: str) -> None:
        """Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname: source file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.load_model(fname)

    def fit(
        self,
        dataset: Union[List[Episode], List[Transition], MDPDataset],
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "tjuOfflineRL_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Trains with the given dataset.

        .. code-block:: python

            algo.fit(episodes, n_steps=1000000)

        Args:
            dataset: list of episodes to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            list of result tuples (epoch, metrics) per epoch.

        """
        results = list(
            self.fitter(
                dataset,
                n_epochs,
                n_steps,
                n_steps_per_epoch,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodes,
                save_interval,
                scorers,
                shuffle,
                callback,
            )
        )
        return results

    def fitter(
        self,
        dataset: Union[List[Episode], List[Transition], MDPDataset],
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "tjuOfflineRL_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
             iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: offline dataset to train.
            n_epochs: the number of epochs to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            shuffle: flag to shuffle transitions on each epoch.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            iterator yielding current epoch and metrics dict.

        """

        transitions = []
        if isinstance(dataset, MDPDataset):
            for episode in dataset.episodes:
                transitions += episode.transitions
        elif not dataset:
            raise ValueError("empty dataset is not supported.")
        elif isinstance(dataset[0], Episode):
            for episode in cast(List[Episode], dataset):
                transitions += episode.transitions
        elif isinstance(dataset[0], Transition):
            transitions = list(cast(List[Transition], dataset))
        else:
            raise ValueError(f"invalid dataset type: {type(dataset)}")

        # check action space
        if self.get_action_type() == ActionSpace.BOTH:
            pass
        elif transitions[0].is_discrete:
            assert (
                self.get_action_type() == ActionSpace.DISCRETE
            ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
        else:
            assert (
                self.get_action_type() == ActionSpace.CONTINUOUS
            ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

        iterator: TransitionIterator
        if n_epochs is None and n_steps is not None:
            assert n_steps >= n_steps_per_epoch
            n_epochs = n_steps // n_steps_per_epoch
            iterator = RandomIterator(
                transitions,
                n_steps_per_epoch,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
            )
            LOG.debug("RandomIterator is selected.")
        elif n_epochs is not None and n_steps is None:
            iterator = RoundIterator(
                transitions,
                batch_size=self._batch_size,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_frames=self._n_frames,
                real_ratio=self._real_ratio,
                generated_maxlen=self._generated_maxlen,
                shuffle=shuffle,
            )
            LOG.debug("RoundIterator is selected.")
        else:
            raise ValueError("Either of n_epochs or n_steps must be given.")

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard_dir,
        )

        # add reference to active logger to algo class during fit
        self._active_logger = logger

        # initialize scaler
        if self._scaler:
            LOG.debug("Fitting scaler...", scaler=self._scaler.get_type())
            self._scaler.fit(transitions)

        # initialize action scaler
        if self._action_scaler:
            LOG.debug(
                "Fitting action scaler...",
                action_scaler=self._action_scaler.get_type(),
            )
            self._action_scaler.fit(transitions)

        # initialize reward scaler
        if self._reward_scaler:
            LOG.debug(
                "Fitting reward scaler...",
                reward_scaler=self._reward_scaler.get_type(),
            )
            self._reward_scaler.fit(transitions)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            transition = iterator.transitions[0]
            action_size = transition.get_action_size()
            observation_shape = tuple(transition.get_observation_shape())
            self.create_impl(
                self._process_observation_shape(observation_shape), action_size
            )
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        self.save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        # training loop
        total_step = 0
        for epoch in range(1, n_epochs + 1):

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(len(iterator)),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            iterator.reset()

            for itr in range_gen:

                # generate new transitions with dynamics models
                new_transitions = self.generate_new_data(
                    transitions=iterator.transitions,
                )
                if new_transitions:
                    iterator.add_generated_transitions(new_transitions)
                    LOG.debug(
                        f"{len(new_transitions)} transitions are generated.",
                        real_transitions=len(iterator.transitions),
                        fake_transitions=len(iterator.generated_transitions),
                    )

                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = next(iterator)

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # save loss to loss history dict
            self._loss_history["epoch"].append(epoch)
            self._loss_history["step"].append(total_step)
            for name, vals in epoch_loss.items():
                if vals:
                    self._loss_history[name].append(np.mean(vals))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger = None

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape: observation shape.
            action_size: dimension of action-space.

        """
        if self._impl:
            LOG.warn("Parameters will be reinitialized.")
        self._create_impl(observation_shape, action_size)

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        raise NotImplementedError

    def build_with_dataset(self, dataset: MDPDataset) -> None:
        """Instantiate implementation object with MDPDataset object.

        Args:
            dataset: dataset.

        """
        observation_shape = dataset.get_observation_shape()
        self.create_impl(
            self._process_observation_shape(observation_shape),
            dataset.get_action_size(),
        )

    def build_with_env(self, env: gym.Env) -> None:
        """Instantiate implementation object with OpenAI Gym object.

        Args:
            env: gym-like environment.

        """
        observation_shape = env.observation_space.shape
        self.create_impl(
            self._process_observation_shape(observation_shape),
            get_action_size_from_env(env),
        )

    def _process_observation_shape(
        self, observation_shape: Sequence[int]
    ) -> Sequence[int]:
        if len(observation_shape) == 3:
            n_channels = observation_shape[0]
            image_size = observation_shape[1:]
            # frame stacking for image observation
            observation_shape = (self._n_frames * n_channels, *image_size)
        return observation_shape

    def update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: mini-batch data.

        Returns:
            dictionary of metrics.

        """
        loss = self._update(batch)
        self._grad_step += 1
        return loss

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        raise NotImplementedError

    def update_redq(self, batch: TransitionMiniBatch, i_update: int) -> Dict[str, float]:
        loss = self._update_redq(batch, i_update)
        self._grad_step += 1
        return loss

    def _update_redq(self, batch: TransitionMiniBatch, i_update: int) -> Dict[str, float]:
        raise NotImplementedError

    def generate_new_data(
        self, transitions: List[Transition]
    ) -> Optional[List[Transition]]:
        """Returns generated transitions for data augmentation.

        This method is for model-based RL algorithms.

        Args:
            transitions: list of transitions.

        Returns:
            list of new transitions.

        """
        return None

    def _prepare_logger(
        self,
        save_metrics: bool,
        experiment_name: Optional[str],
        with_timestamp: bool,
        logdir: str,
        verbose: bool,
        tensorboard_dir: Optional[str],
    ) -> TJUOfflineRLLogger:
        if experiment_name is None:
            experiment_name = self.__class__.__name__

        logger = TJUOfflineRLLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            with_timestamp=with_timestamp,
        )

        return logger

    def _evaluate(
        self,
        episodes: List[Episode],
        scorers: Dict[str, Callable[[Any, List[Episode]], float]],
        logger: TJUOfflineRLLogger,
    ) -> None:
        for name, scorer in scorers.items():
            # evaluation with test data
            test_score = scorer(self, episodes)

            # logging metrics
            logger.add_metric(name, test_score)

            # store metric locally
            if test_score is not None:
                self._eval_results[name].append(test_score)

    def save_params(self, logger: TJUOfflineRLLogger) -> None:
        """Saves configurations as params.json.

        Args:
            logger: logger object.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        # get hyperparameters without impl
        params = {}
        with disable_parallel():
            for k, v in self.get_params(deep=False).items():
                if isinstance(v, (ImplBase, LearnableBase)):
                    continue
                params[k] = v

        # save algorithm name
        params["algorithm"] = self.__class__.__name__

        # save shapes
        params["observation_shape"] = self._impl.observation_shape
        params["action_size"] = self._impl.action_size

        # serialize objects
        params = _serialize_params(params)

        logger.add_params(params)

    def get_action_type(self) -> ActionSpace:
        """Returns action type (continuous or discrete).

        Returns:
            action type.

        """
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        """Batch size to train.

        Returns:
            int: batch size.

        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    @property
    def n_frames(self) -> int:
        """Number of frames to stack.

        This is only for image observation.

        Returns:
            int: number of frames to stack.

        """
        return self._n_frames

    @n_frames.setter
    def n_frames(self, n_frames: int) -> None:
        self._n_frames = n_frames

    @property
    def n_steps(self) -> int:
        """N-step TD backup.

        Returns:
            int: N-step TD backup.

        """
        return self._n_steps

    @n_steps.setter
    def n_steps(self, n_steps: int) -> None:
        self._n_steps = n_steps

    @property
    def gamma(self) -> float:
        """Discount factor.

        Returns:
            float: discount factor.

        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        self._gamma = gamma

    @property
    def scaler(self) -> Optional[Scaler]:
        """Preprocessing scaler.

        Returns:
            Optional[Scaler]: preprocessing scaler.

        """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler: Scaler) -> None:
        self._scaler = scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        """Preprocessing action scaler.

        Returns:
            Optional[ActionScaler]: preprocessing action scaler.

        """
        return self._action_scaler

    @action_scaler.setter
    def action_scaler(self, action_scaler: ActionScaler) -> None:
        self._action_scaler = action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        """Preprocessing reward scaler.

        Returns:
            Optional[RewardScaler]: preprocessing reward scaler.

        """
        return self._reward_scaler

    @reward_scaler.setter
    def reward_scaler(self, reward_scaler: RewardScaler) -> None:
        self._reward_scaler = reward_scaler

    @property
    def impl(self) -> Optional[ImplBase]:
        """Implementation object.

        Returns:
            Optional[ImplBase]: implementation object.

        """
        return self._impl

    @impl.setter
    def impl(self, impl: ImplBase) -> None:
        self._impl = impl

    @property
    def observation_shape(self) -> Optional[Sequence[int]]:
        """Observation shape.

        Returns:
            Optional[Sequence[int]]: observation shape.

        """
        if self._impl:
            return self._impl.observation_shape
        return None

    @property
    def action_size(self) -> Optional[int]:
        """Action size.

        Returns:
            Optional[int]: action size.

        """
        if self._impl:
            return self._impl.action_size
        return None

    @property
    def active_logger(self) -> Optional[TJUOfflineRLLogger]:
        """Active TJUOfflineRLLogger object.

        This will be only available during training.

        Returns:
            logger object.

        """
        return self._active_logger

    def set_active_logger(self, logger: TJUOfflineRLLogger) -> None:
        """Set active TJUOfflineRLLogger object

        Args:
            logger: logger object.

        """
        self._active_logger = logger

    @property
    def grad_step(self) -> int:
        """Total gradient step counter.

        This value will keep counting after ``fit`` and ``fit_online``
        methods finish.

        Returns:
            total gradient step counter.

        """
        return self._grad_step

    def set_grad_step(self, grad_step: int) -> None:
        """Set total gradient step counter.

        This method can be used to restart the middle of training with an
        arbitrary gradient step counter, which has effects on periodic
        functions such as the target update.

        Args:
            grad_step: total gradient step counter.

        """
        self._grad_step = grad_step
