# Combine ED2 with Dreamer
In this project, we mainly follow the source code from Dreamer: https://github.com/danijar/dreamer


## Instructions

Get dependencies (the same with Dreamer):
```
pip3 install --user tensorflow-gpu==2.1.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib
```

Training:
```
python3 run.py
```

Result visualizing:
```
tensorboard --logdir ./logdir
```
