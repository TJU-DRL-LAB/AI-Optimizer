import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
import argparse
import seaborn as sns
import csv


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, nargs='+', default=(0,),
                        help='random seed (default: (0,))')
    parser.add_argument('--max_m', type=int, default=None,
                        help='maximum million')
    parser.add_argument('--smooth_coeff', type=int, default=25,
                        help='smooth coeff')
    parser.add_argument('--env_name', type=str, default='mt10',
                        help='environment trained on (default: mt10)')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='directory for tensorboard logs (default: ./log)')
    parser.add_argument( "--id", type=str, nargs='+', default=('origin',),
                        help="id for tensorboard")
    parser.add_argument( "--tags", type=str, nargs='+', default=None,
                        help="id for tensorboard")
    parser.add_argument('--output_dir', type=str, default='./fig',
                        help='directory for plot output (default: ./fig)')
    parser.add_argument('--entry', type=str, default='Running_Average_Rewards',
                        help='Record Entry')
    parser.add_argument('--add_tag', type=str, default='',
                        help='added tag')
    args = parser.parse_args()
    return args


args = get_args()
env_name = args.env_name
env_id = args.id

if args.tags is None:
    args.tags = args.id
assert len(args.tags) == len(args.id)


def post_process(array):
    smoth_para = args.smooth_coeff
    new_array = []
    for i in range(len(array)):
        if i < len(array) - smoth_para:
            new_array.append(np.mean(array[i:i+smoth_para]))
        else:
            new_array.append(np.mean(array[i:None]))
    return new_array    


sns.set("paper")

current_palette = sns.color_palette()
sns.palplot(current_palette)

fig = plt.figure(figsize=(14,7))
plt.subplots_adjust(left=0.07, bottom=0.15, right=1, top=0.90,
                wspace=0, hspace=0)

ax1 = fig.add_subplot(111)
colors = current_palette
linestyles_choose = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

for eachcolor, eachlinestyle, exp_name, exp_tag in zip(colors, linestyles_choose, args.id, args.tags ):
    min_step_number = 1000000000000
    step_number = []
    all_scores = {}

    for seed in args.seed:
        file_path = os.path.join(args.log_dir, exp_name, env_name, str(seed), 'progress.csv')

        all_scores[seed] = []
        temp_step_number = []
        with open(file_path,'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                all_scores[seed].append(float(row[args.entry]))
                # temp_step_number.append(int(row["Total Frames"]))
                temp_step_number.append(int(row["Epoch"]))

        if temp_step_number[-1] < min_step_number:
            min_step_number = temp_step_number[-1]
            step_number = temp_step_number 

    all_mean = []
    all_upper = []
    all_lower = []

    # step_number = np.array(step_number) / 1e6
    step_number = np.array(step_number)
    final_step = []
    for i in range(len(step_number)):
        if args.max_m is not None and step_number[i] >= args.max_m:
            continue
        final_step.append(step_number[i])
        temp_list = []
        for key, valueList in all_scores.items():
            try: 
                temp_list.append(valueList[i])
            except Exception:
                print(i)
                # exit()
        all_mean.append(np.mean(temp_list))
        all_upper.append(np.mean(temp_list) + np.std(temp_list))
        all_lower.append(np.mean(temp_list) - np.std(temp_list))
    # print(exp_tag, np.mean(all_mean[-10:]))
    all_mean = post_process(all_mean)
    all_lower = post_process(all_lower)
    all_upper = post_process(all_upper)

    ax1.plot(final_step, all_mean, label=exp_tag, color=eachcolor, linestyle=eachlinestyle, linewidth=2)
    ax1.plot(final_step, all_upper, color=eachcolor, linestyle=eachlinestyle, alpha = 0.23, linewidth=1)
    ax1.plot(final_step, all_lower, color=eachcolor, linestyle=eachlinestyle, alpha = 0.23, linewidth=1)
    ax1.fill_between(final_step, all_lower, all_upper, alpha=0.2, color=eachcolor)


ax1.set_xlabel('Epochs', fontsize=30)
ax1.tick_params(labelsize=25)

box = ax1.get_position()

leg = ax1.legend(
           loc='best',
           ncol=1,
           fontsize=25)

for legobj in leg.legendHandles:
    legobj.set_linewidth(10.0)

plt.title("{} {}".format(env_name, args.entry), fontsize=40)
if not os.path.exists( args.output_dir ):
    os.mkdir( args.output_dir )
plt.savefig( os.path.join( args.output_dir, '{}_{}{}.png'.format(env_name, args.entry, args.add_tag) ) )
plt.close()