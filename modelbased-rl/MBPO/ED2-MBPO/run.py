import os
task = 'halfcheetah'
model = '0'
if not os.path.exists('log_files/' + task + '/' + model + '/'):
    os.makedirs('log_files/' + task + '/' + model + '/')
log = 0
for gpu in range(5):
    comand = 'CUDA_VISIBLE_DEVICES={gpu} nohup python -u mbpo.py --config=examples.config.{task}.{model} --gpus=1 --trial-gpus=1 --cpus=1 --trial-cpus=1 > ./log_files/{task}/{model}/{log}.log 2>&1 &'.format(model=model, task=task, gpu=gpu, log=log)
    os.system(comand)
    log += 1
