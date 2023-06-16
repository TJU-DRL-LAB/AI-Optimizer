import torch

from ..utils.util import DEFAULT_DEVICE, extract_sub_dict
from .iql import IQL

class IQL_online(IQL):
    def __init__(self, critic, vf, policy, optimizer_ctor,
                 tau, beta, discount, target_update_rate, ckpt_path, copy_to_target=True):

        super().__init__(critic=critic, vf=vf, policy=policy,
                         optimizer_ctor=optimizer_ctor,
                         max_steps=None,
                         tau=tau, beta=beta,
                         discount=discount,
                         target_update_rate=target_update_rate,
                         use_lr_scheduler=False)

        # load checkpoint if ckpt_path is not None
        if ckpt_path is not None:

            map_location = None
            if not torch.cuda.is_available():
                map_location = torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            # extract sub-dictionary
            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic", checkpoint)

            self.policy.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                target_critic_state_dict = extract_sub_dict("target_critic", checkpoint)
                self.target_critic.load_state_dict(target_critic_state_dict)

            self.vf.load_state_dict(extract_sub_dict("vf", checkpoint))


