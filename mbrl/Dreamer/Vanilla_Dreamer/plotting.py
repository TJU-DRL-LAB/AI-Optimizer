import argparse
import collections
import functools
import json
import multiprocessing as mp
import pathlib
import re
import subprocess

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

Run = collections.namedtuple('Run', 'task method seed xs ys color')

PALETTE = 10 * (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    '#fdbf6f')

LEGEND = dict(
    fontsize='medium', numpoints=1, labelspacing=0, columnspacing=1.2,
    handlelength=1.5, handletextpad=0.5, ncol=4, loc='lower center')


def find_keys(args):
  filename = next(args.indir[0].glob('**/*.jsonl'))
  keys = set()
  for line in filename.read_text().split('\n'):
    if line:
      keys |= json.loads(line).keys()
  print(f'Keys ({len(keys)}):', ', '.join(keys), flush=True)


def load_runs(args):
  toload = []
  for indir in args.indir:
    filenames = list(indir.glob('**/*.jsonl'))
    for filename in filenames:
      task, method, seed = filename.relative_to(indir).parts[:-1]
      if not any(p.search(task) for p in args.tasks):
        continue
      if not any(p.search(method) for p in args.methods):
        continue
      if method not in args.colors:
        args.colors[method] = args.palette[len(args.colors)]
      toload.append((filename, indir))
  print(f'Loading {len(toload)} of {len(filenames)} runs...')
  jobs = [functools.partial(load_run, f, i, args) for f, i in toload]
  with mp.Pool(10) as pool:
    promises = [pool.apply_async(j) for j in jobs]
    runs = [p.get() for p in promises]
  runs = [r for r in runs if r is not None]
  return runs


def load_run(filename, indir, args):
  task, method, seed = filename.relative_to(indir).parts[:-1]
  try:
    # Future pandas releases will support JSON files with NaN values.
    # df = pd.read_json(filename, lines=True)
    with filename.open() as f:
      df = pd.DataFrame([json.loads(l) for l in f.readlines()])
  except ValueError as e:
    print('Invalid', filename.relative_to(indir), e)
    return
  try:
    df = df[[args.xaxis, args.yaxis]].dropna()
  except KeyError:
    return
  xs = df[args.xaxis].to_numpy()
  ys = df[args.yaxis].to_numpy()
  color = args.colors[method]
  return Run(task, method, seed, xs, ys, color)


def load_baselines(args):
  runs = []
  directory = pathlib.Path(__file__).parent / 'baselines'
  for filename in directory.glob('**/*.json'):
    for task, methods in json.loads(filename.read_text()).items():
      for method, score in methods.items():
        if not any(p.search(method) for p in args.baselines):
          continue
        if method not in args.colors:
          args.colors[method] = args.palette[len(args.colors)]
        color = args.colors[method]
        runs.append(Run(task, method, None, None, score, color))
  return runs


def stats(runs):
  baselines = sorted(set(r.method for r in runs if r.xs is None))
  runs = [r for r in runs if r.xs is not None]
  tasks = sorted(set(r.task for r in runs))
  methods = sorted(set(r.method for r in runs))
  seeds = sorted(set(r.seed for r in runs))
  print('Loaded', len(runs), 'runs.')
  print(f'Tasks     ({len(tasks)}):', ', '.join(tasks))
  print(f'Methods   ({len(methods)}):', ', '.join(methods))
  print(f'Seeds     ({len(seeds)}):', ', '.join(seeds))
  print(f'Baselines ({len(baselines)}):', ', '.join(baselines))


def figure(runs, args):
  tasks = sorted(set(r.task for r in runs if r.xs is not None))
  rows = int(np.ceil(len(tasks) / args.cols))
  figsize = args.size[0] * args.cols, args.size[1] * rows
  fig, axes = plt.subplots(rows, args.cols, figsize=figsize)
  for task, ax in zip(tasks, axes.flatten()):
    relevant = [r for r in runs if r.task == task]
    plot(task, ax, relevant, args)
  if args.xlim:
    for ax in axes[:-1].flatten():
      ax.xaxis.get_offset_text().set_visible(False)
  if args.xlabel:
    for ax in axes[-1]:
      ax.set_xlabel(args.xlabel)
  if args.ylabel:
    for ax in axes[:, 0]:
      ax.set_ylabel(args.ylabel)
  for ax in axes[len(tasks):]:
    ax.axis('off')
  legend(fig, args.labels, **LEGEND)
  return fig


def plot(task, ax, runs, args):
  try:
    title = task.split('_', 1)[1].replace('_', ' ').title()
  except IndexError:
    title = task.title()
  ax.set_title(title)
  methods = []
  methods += sorted(set(r.method for r in runs if r.xs is not None))
  methods += sorted(set(r.method for r in runs if r.xs is None))
  xlim = [+np.inf, -np.inf]
  for index, method in enumerate(methods):
    relevant = [r for r in runs if r.method == method]
    if not relevant:
      continue
    if any(r.xs is None for r in relevant):
      baseline(index, method, ax, relevant, args)
    else:
      if args.aggregate == 'std':
        xs, ys = curve_std(index, method, ax, relevant, args)
      elif args.aggregate == 'none':
        xs, ys = curve_individual(index, method, ax, relevant, args)
      else:
        raise NotImplementedError(args.aggregate)
      xlim = [min(xlim[0], xs.min()), max(xlim[1], xs.max())]
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  steps = [1, 2, 2.5, 5, 10]
  ax.xaxis.set_major_locator(ticker.MaxNLocator(args.xticks, steps=steps))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(args.yticks, steps=steps))
  ax.set_xlim(args.xlim or xlim)
  if args.xlim:
    ticks = sorted({*ax.get_xticks(), *args.xlim})
    ticks = [x for x in ticks if args.xlim[0] <= x <= args.xlim[1]]
    ax.set_xticks(ticks)
  if args.ylim:
    ax.set_ylim(args.ylim)
    ticks = sorted({*ax.get_yticks(), *args.ylim})
    ticks = [x for x in ticks if args.ylim[0] <= x <= args.ylim[1]]
    ax.set_yticks(ticks)


def curve_individual(index, method, ax, runs, args):
  if args.bins:
    for index, run in enumerate(runs):
      xs, ys = binning(run.xs, run.ys, args.bins, np.nanmean)
      runs[index] = run._replace(xs=xs, ys=ys)
  zorder = 10000 - 10 * index - 1
  for run in runs:
    ax.plot(run.xs, run.ys, label=method, color=run.color, zorder=zorder)
  return runs[0].xs, runs[0].ys


def curve_std(index, method, ax, runs, args):
  if args.bins:
    for index, run in enumerate(runs):
      xs, ys = binning(run.xs, run.ys, args.bins, np.nanmean)
      runs[index] = run._replace(xs=xs, ys=ys)
  xs = np.concatenate([r.xs for r in runs])
  ys = np.concatenate([r.ys for r in runs])
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  color = runs[0].color
  if args.bins:
    reducer = lambda y: (np.nanmean(np.array(y)), np.nanstd(np.array(y)))
    xs, ys = binning(xs, ys, args.bins, reducer)
    ys, std = ys.T
    kw = dict(color=color, zorder=10000 - 10 * index, alpha=0.1, linewidths=0)
    ax.fill_between(xs, ys - std, ys + std, **kw)
  ax.plot(xs, ys, label=method, color=color, zorder=10000 - 10 * index - 1)
  return xs, ys


def baseline(index, method, ax, runs, args):
  assert len(runs) == 1 and runs[0].xs is None
  y = np.mean(runs[0].ys)
  kw = dict(ls='--', color=runs[0].color, zorder=5000 - 10 * index - 1)
  ax.axhline(y, label=method, **kw)


def binning(xs, ys, bins, reducer):
  binned_xs = np.arange(xs.min(), xs.max() + 1e-10, bins)
  binned_ys = []
  for start, stop in zip([-np.inf] + list(binned_xs), binned_xs):
    left = (xs <= start).sum()
    right = (xs <= stop).sum()
    binned_ys.append(reducer(ys[left:right]))
  binned_ys = np.array(binned_ys)
  return binned_xs, binned_ys


def legend(fig, mapping=None, **kwargs):
  entries = {}
  for ax in fig.axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
      if mapping and label in mapping:
        label = mapping[label]
      entries[label] = handle
  leg = fig.legend(entries.values(), entries.keys(), **kwargs)
  leg.get_frame().set_edgecolor('white')
  extent = leg.get_window_extent(fig.canvas.get_renderer())
  extent = extent.transformed(fig.transFigure.inverted())
  yloc, xloc = kwargs['loc'].split()
  y0 = dict(lower=extent.y1, center=0, upper=0)[yloc]
  y1 = dict(lower=1, center=1, upper=extent.y0)[yloc]
  x0 = dict(left=extent.x1, center=0, right=0)[xloc]
  x1 = dict(left=1, center=1, right=extent.x0)[xloc]
  fig.tight_layout(rect=[x0, y0, x1, y1], h_pad=0.5, w_pad=0.5)


def save(fig, args):
  args.outdir.mkdir(parents=True, exist_ok=True)
  filename = args.outdir / 'curves.png'
  fig.savefig(filename, dpi=130)
  print('Saved to', filename)
  filename = args.outdir / 'curves.pdf'
  fig.savefig(filename)
  try:
    subprocess.call(['pdfcrop', str(filename), str(filename)])
  except FileNotFoundError:
    pass  # Install texlive-extra-utils.


def main(args):
  find_keys(args)
  runs = load_runs(args) + load_baselines(args)
  stats(runs)
  if not runs:
    print('Noting to plot.')
    return
  print('Plotting...')
  fig = figure(runs, args)
  save(fig, args)


def parse_args():
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--subdir', type=boolean, default=True)
  parser.add_argument('--xaxis', type=str, required=True)
  parser.add_argument('--yaxis', type=str, required=True)
  parser.add_argument('--tasks', nargs='+', default=[r'.*'])
  parser.add_argument('--methods', nargs='+', default=[r'.*'])
  parser.add_argument('--baselines', nargs='+', default=[])
  parser.add_argument('--bins', type=float, default=0)
  parser.add_argument('--aggregate', type=str, default='std')
  parser.add_argument('--size', nargs=2, type=float, default=[2.5, 2.3])
  parser.add_argument('--cols', type=int, default=4)
  parser.add_argument('--xlim', nargs=2, type=float, default=None)
  parser.add_argument('--ylim', nargs=2, type=float, default=None)
  parser.add_argument('--xlabel', type=str, default=None)
  parser.add_argument('--ylabel', type=str, default=None)
  parser.add_argument('--xticks', type=int, default=6)
  parser.add_argument('--yticks', type=int, default=5)
  parser.add_argument('--labels', nargs='+', default=None)
  parser.add_argument('--palette', nargs='+', default=PALETTE)
  parser.add_argument('--colors', nargs='+', default={})
  args = parser.parse_args()
  if args.subdir:
    args.outdir /= args.indir[0].stem
  args.indir = [d.expanduser() for d in args.indir]
  args.outdir = args.outdir.expanduser()
  if args.labels:
    assert len(args.labels) % 2 == 0
    args.labels = {k: v for k, v in zip(args.labels[:-1], args.labels[1:])}
  if args.colors:
    assert len(args.colors) % 2 == 0
    args.colors = {k: v for k, v in zip(args.colors[:-1], args.colors[1:])}
  args.tasks = [re.compile(p) for p in args.tasks]
  args.methods = [re.compile(p) for p in args.methods]
  args.baselines = [re.compile(p) for p in args.baselines]
  args.palette = 10 * args.palette
  return args


if __name__ == '__main__':
  main(parse_args())
