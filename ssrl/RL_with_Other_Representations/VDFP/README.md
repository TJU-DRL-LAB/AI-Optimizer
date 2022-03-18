# AAAI2021-VDFP

Source code and raw data of learning curves for AAAI 2021 paper - 《Foresee then Evaluate: Decomposing Value Estimation with Latent Future Prediction》

## *To-Do

-  ArXiv version will be available soon.
-  Codes for VD-DQN, n-step DDPG will be added soon.
-  ...


## A. Description  

The source code mainly contains:  
-  implementation of our algorithm (VDFP) and other benchmark algorithms used in our experiments;  
-  the raw learning curves data and plot code.  

All the implementation and experimental details mentioned in our paper and the Supplementary Material can be found in our codes.  
  
  
## B. Environment Setup  

We conduct our experiments on MuJoCo continuous control tasks in OpenAI gym. Our codes are implemented with **Python 3.6** and **Tensorflow 1.8**.  
We run our experiments on both **Windows 7** and **Ubuntu 16.04 LTS** operating systems.  
  
Main dependencies and versions are listed below:  

| Dependency | Version |
| ------ | ------ |
| gym | 0.9.1 |
| mujoco-py | 0.5.7 | 
| mjpro | mjpro131 | 
| tensorflow | 1.8.0 | 
| tensorboard | 1.8.0 |
| scipy | 1.2.1 | 
| scikit-learn | 0.20.3 | 
| matplotlib | 3.0.3 | 
  
  
  
## C. Examples  
  
Examples of run commands can be seen in the file below:
> ./run/run_vdfp.sh

