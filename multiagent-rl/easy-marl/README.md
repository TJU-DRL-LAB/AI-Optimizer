# Easy-MARL  
A very easy marl library for beginners.  


## Algorithms  
    DQN-based (for discrete environment):  
        IDQN, VDN, QMIX  
    DDPG-based (for continuous environment):  
        IDDPG, MADDPG  
    PPO-based (for both discrete and continuous environment):   
        IPPO, MAPPO  



## Environments  
    discrete_meeting  
    discrete_magym  
    continuous_meeting  
    continuous_mpe  


## Run  
    python main_dqn.py --agent-name IDQN --env-name discrete_meeting  
    python main_dqn.py --agent-name VDN --env-name discrete_magym --scenario-name Switch4-v0
    python main_ddpg.py --agent-name IDDPG --env-name continuous_meeting  
    python main_ddpg.py --agent-name MADDPG --env-name continuous_mpe --scenario-name simple_spread  
    python main_ppo.py --agent-name IPPO --env-name discrete_meeting  
    python main_ppo.py --agent-name MAPPO --env-name discrete_magym --scenario-name Combat-v0  
    python main_ppo.py --agent-name IPPO --env-name continuous_meeting 
    python main_ppo.py --agent-name MAPPO --env-name continuous_mpe --scenario-name simple_tag  