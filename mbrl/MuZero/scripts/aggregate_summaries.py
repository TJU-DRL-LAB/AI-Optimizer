import os
from tensorboard.backend.event_processing import event_accumulator as ea
from torch.utils.tensorboard import SummaryWriter


def aggregate_summaries(logdir: str, exp_path: str, ):
    # we recognize all files which have tfevents
    scalars_info = {}
    for root, dirs, files in os.walk(logdir):
        for event_file in [x for x in files if 'tfevents' in x]:
            event_path = os.path.join(root, event_file)

            acc = ea.EventAccumulator(event_path)
            acc.Reload()

            # only support scalar now
            scalar_list = acc.Tags()['scalars']
            for tag in scalar_list:
                if tag not in scalars_info:
                    scalars_info[tag] = {'data': {}}
                for s in acc.Scalars(tag):
                    if s.step not in scalars_info[tag]['data']:
                        scalars_info[tag]['data'][s.step] = [s.value]
                    else:
                        scalars_info[tag]['data'][s.step].append(s.value)

    summary_writer = SummaryWriter(exp_path, flush_secs=10)

    for tag in scalars_info:
        for steps in sorted(list(scalars_info[tag]['data'].keys())):
            values = scalars_info[tag]['data'][steps]
            summary_writer.add_scalar(tag, sum(values) / len(values), steps)
        summary_writer.flush()

    summary_writer.close()


if __name__ == '__main__':
    """Aggregates multiple runs of the each configuration"""
    result_path = '../results'
    base_aggregate_path = '../aggregate_results'
    for root, dirs, files in os.walk(result_path):
        if len(dirs) > 0 and 'seed' in dirs[0]:
            print(root, dirs, files)
            aggregate_path = base_aggregate_path + root.split(result_path)[1]
            os.makedirs(aggregate_path, exist_ok=True)
            aggregate_summaries(root, aggregate_path)
