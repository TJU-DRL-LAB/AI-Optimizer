# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf

from planet import tools


_Phase = collections.namedtuple(
    'Phase',
    'name, writer, op, batch_size, steps, feed, report_every, log_every,'
    'checkpoint_every, restore_every')


class Trainer(object):
  """Execute operations in a trainer and coordinate logging and checkpoints.

  Supports multiple phases, that define their own operations to run, and
  intervals for reporting scores, logging summaries, and storing checkpoints.
  All class state is stored in-graph to properly recover from checkpoints.
  """

  def __init__(self, logdir, config=None):
    """Execute operations in a trainer and coordinate logging and checkpoints.

    The `reset` property is used to indicate switching to a new phase, so that
    the model can start a new computation in case its computation is split over
    multiple training steps.

    Args:
      logdir: Will contain checkpoints and summaries for each phase.
      config: configuration AttrDict.
    """
    self._logdir = logdir
    self._global_step = tf.train.get_or_create_global_step()
    self._step = tf.placeholder(tf.int32, name='step')
    self._phase = tf.placeholder(tf.string, name='phase')
    self._log = tf.placeholder(tf.bool, name='log')
    self._report = tf.placeholder(tf.bool, name='report')
    self._reset = tf.placeholder(tf.bool, name='reset')
    self._phases = []
    # Checkpointing.
    self._loaders = []
    self._savers = []
    self._logdirs = []
    self._checkpoints = []
    self._config = config or tools.AttrDict()

  @property
  def global_step(self):
    """Global number of steps performed over all phases."""
    return self._global_step

  @property
  def step(self):
    """Number of steps performed in the current phase."""
    return self._step

  @property
  def phase(self):
    """Name of the current training phase."""
    return self._phase

  @property
  def log(self):
    """Whether the model should compute summaries."""
    return self._log

  @property
  def reset(self):
    """Whether the model should reset its state."""
    return self._reset

  def add_saver(
      self, include=r'.*', exclude=r'.^', logdir=None, load=True, save=True,
      checkpoint=None):
    """Add a saver to save or load variables.

    Args:
      include: One or more regexes to match variable names to include.
      exclude: One or more regexes to match variable names to exclude.
      logdir: Directory for saver to store and search for checkpoints.
      load: Whether to use the saver to restore variables.
      save: Whether to use the saver to save variables.
      checkpoint: Checkpoint name to load; None for newest.
    """
    variables = tools.filter_variables(include, exclude)
    saver = tf.train.Saver(variables, keep_checkpoint_every_n_hours=2)
    if load:
      self._loaders.append(saver)
    if save:
      self._savers.append(saver)
    self._logdirs.append(logdir or self._logdir)
    if checkpoint is None and self._config.checkpoint_to_load:
      self._checkpoints.append(
          os.path.join(self._logdirs[-1], self._config.checkpoint_to_load))
    else:
      self._checkpoints.append(checkpoint)

  def add_phase(
      self, name, steps, score, summary, batch_size=1,
      report_every=None, log_every=None, checkpoint_every=None,
      restore_every=None, feed=None):
    """Add a phase to the trainer protocol.

    The score tensor can either be a scalar or vector, to support single and
    batched computations.

    Args:
      name: Name for the phase, used for the summary writer.
      steps: Duration of the phase in steps.
      score: Tensor holding the current scores.
      summary: Tensor holding summary string to write if not an empty string.
      batch_size: Increment size of the global step.
      report_every: Yield mean score every this number of steps.
      log_every: Request summaries via `log` tensor every this number of steps.
      checkpoint_every: Write checkpoint every this number of steps.
      restore_every: Restore from the latest checkpoint every this many steps.
      feed: Additional feed dictionary for the session run call.
    """
    score = tf.convert_to_tensor(score, tf.float32)
    summary = tf.convert_to_tensor(summary, tf.string)
    feed = feed or {}
    if not score.shape.ndims:
      score = score[None]
    writer = self._logdir and tf.summary.FileWriter(
        os.path.join(self._logdir, name),
        tf.get_default_graph(), flush_secs=30)
    op = self._define_step(name, batch_size, score, summary)
    self._phases.append(_Phase(
        name, writer, op, batch_size, int(steps), feed, report_every,
        log_every, checkpoint_every, restore_every))

  def run(self, max_step=None, sess=None, unused_saver=None):
    """Run the schedule for a specified number of steps and log scores.

    Args:
      max_step: Run the operations until the step reaches this limit.
      sess: Session to use to run the phase operation.
    """
    for _ in self.iterate(max_step, sess):
      pass

  def iterate(self, max_step=None, sess=None):
    """Run the schedule for a specified number of steps and yield scores.

    Call the operation of the current phase until the global step reaches the
    specified maximum step. Phases are repeated over and over in the order they
    were added.

    Args:
      max_step: Run the operations until the step reaches this limit.
      sess: Session to use to run the phase operation.

    Yields:
      Reported mean scores.
    """
    sess = sess or self._create_session()
    with sess:
      self._initialize_variables(
          sess, self._loaders, self._logdirs, self._checkpoints)
      sess.graph.finalize()
      while True:
        global_step = sess.run(self._global_step)
        if max_step and global_step >= max_step:
          break
        phase, epoch, steps_in = self._find_current_phase(global_step)
        phase_step = epoch * phase.steps + steps_in
        if steps_in % phase.steps < phase.batch_size:
          message = '\n' + ('-' * 50) + '\n'
          message += 'Epoch {} phase {} (phase step {}, global step {}).'
          tf.logging.info(message.format(
              epoch + 1, phase.name, phase_step, global_step))
        # Populate book keeping tensors.
        phase.feed[self._step] = phase_step
        phase.feed[self._phase] = phase.name
        phase.feed[self._reset] = (steps_in < phase.batch_size)
        phase.feed[self._log] = phase.writer and self._is_every_steps(
            phase_step, phase.batch_size, phase.log_every)
        phase.feed[self._report] = self._is_every_steps(
            phase_step, phase.batch_size, phase.report_every)
        summary, mean_score, global_step = sess.run(phase.op, phase.feed)
        if self._is_every_steps(
            phase_step, phase.batch_size, phase.checkpoint_every):
          for saver in self._savers:
            self._store_checkpoint(sess, saver, global_step)
        if self._is_every_steps(
            phase_step, phase.batch_size, phase.report_every):
          tf.logging.info('Score {}.'.format(mean_score))
          yield mean_score
        if summary and phase.writer:
          # We want smaller phases to catch up at the beginnig of each epoch so
          # that their graphs are aligned.
          longest_phase = max(phase_.steps for phase_ in self._phases)
          summary_step = epoch * longest_phase + steps_in
          phase.writer.add_summary(summary, summary_step)
        if self._is_every_steps(
            phase_step, phase.batch_size, phase.restore_every):
          self._initialize_variables(
              sess, self._loaders, self._logdirs, self._checkpoints)

  def _is_every_steps(self, phase_step, batch, every):
    """Determine whether a periodic event should happen at this step.

    Args:
      phase_step: The incrementing step.
      batch: The number of steps progressed at once.
      every: The interval of the period.

    Returns:
      Boolean of whether the event should happen.
    """
    if not every:
      return False
    covered_steps = range(phase_step, phase_step + batch)
    return any((step + 1) % every == 0 for step in covered_steps)

  def _find_current_phase(self, global_step):
    """Determine the current phase based on the global step.

    This ensures continuing the correct phase after restoring checkoints.

    Args:
      global_step: The global number of steps performed across all phases.

    Returns:
      Tuple of phase object, epoch number, and phase steps within the epoch.
    """
    epoch_size = sum(phase.steps for phase in self._phases)
    epoch = int(global_step // epoch_size)
    steps_in = global_step % epoch_size
    for phase in self._phases:
      if steps_in < phase.steps:
        return phase, epoch, steps_in
      steps_in -= phase.steps

  def _define_step(self, name, batch_size, score, summary):
    """Combine operations of a phase.

    Keeps track of the mean score and when to report it.

    Args:
      name: Name of the phase used for the score summary.
      batch_size: Increment size of the global step.
      score: Tensor holding the current scores.
      summary: Tensor holding summary string to write if not an empty string.

    Returns:
      Tuple of summary tensor, mean score, and new global step. The mean score
      is zero for non reporting steps.
    """
    with tf.variable_scope('phase_{}'.format(name)):
      score_mean = tools.StreamingMean((), tf.float32, 'score_mean')
      score.set_shape((None,))
      with tf.control_dependencies([score, summary]):
        submit_score = score_mean.submit(score)
      with tf.control_dependencies([submit_score]):
        mean_score = tf.cond(self._report, score_mean.clear, float)
        summary = tf.cond(
            self._report,
            lambda: tf.summary.merge([summary, tf.summary.scalar(
                name + '/score', mean_score, family='trainer')]),
            lambda: summary)
        next_step = self._global_step.assign_add(batch_size)
      with tf.control_dependencies([summary, mean_score, next_step]):
        return (
            tf.identity(summary),
            tf.identity(mean_score),
            tf.identity(next_step))

  def _create_session(self):
    """Create a TensorFlow session with sensible default parameters.

    Returns:
      Session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    try:
      return tf.Session('local', config=config)
    except tf.errors.NotFoundError:
      return tf.Session(config=config)

  def _initialize_variables(self, sess, savers, logdirs, checkpoints):
    """Initialize or restore variables from a checkpoint if available.

    Args:
      sess: Session to initialize variables in.
      savers: List of savers to restore variables.
      logdirs: List of directories for each saver to search for checkpoints.
      checkpoints: List of checkpoint names for each saver; None for newest.
    """
    sess.run(tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()))
    assert len(savers) == len(logdirs) == len(checkpoints)
    for i, (saver, logdir, checkpoint) in enumerate(
        zip(savers, logdirs, checkpoints)):
      logdir = os.path.expanduser(logdir)
      state = tf.train.get_checkpoint_state(logdir)
      if checkpoint:
        checkpoint = os.path.join(logdir, checkpoint)
      if not checkpoint and state and state.model_checkpoint_path:
        checkpoint = state.model_checkpoint_path
      if checkpoint:
        saver.restore(sess, checkpoint)

  def _store_checkpoint(self, sess, saver, global_step):
    """Store a checkpoint if a log directory was provided to the constructor.

    The directory will be created if needed.

    Args:
      sess: Session containing variables to store.
      saver: Saver used for checkpointing.
      global_step: Step number of the checkpoint name.
    """
    if not self._logdir or not saver:
      return
    tf.gfile.MakeDirs(self._logdir)
    filename = os.path.join(self._logdir, 'model.ckpt')
    saver.save(sess, filename, global_step)
