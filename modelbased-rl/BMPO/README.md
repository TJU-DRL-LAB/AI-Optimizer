# Bidirectional Model-based Policy Optimization

This is the TensorFlow implementation for the paper [
Bidrectional Model-based Policy Optimization](https://arxiv.org/abs/2007.01995).


## Requirements
```
pip install -r requirements.txt
```

## Run
```
python main.py --config=config.hopperNT
```
To change hyper-parameters, please modify the corresponding config file in [`config/`](/config).

## Acknowledgments
This code is mainly modified based on the [mbpo](https://github.com/JannerM/mbpo) codebase.
