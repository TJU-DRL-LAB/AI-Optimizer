import numpy as np

from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, explore_policy, max_path_length):
        self.env = env
        self.policy = policy
        self.explore_policy = explore_policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1, explore=False, context=None, infer=False):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        explore_policy = MakeDeterministic(self.explore_policy) if deterministic else self.explore_policy
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        n_success_num = 0
        success = -1
        while n_steps_total < max_samples and n_trajs < max_trajs:

            path = rollout(
                    self.env, policy, explore_policy, max_path_length=self.max_path_length, accum_context=accum_context,
                    explore=explore, context=context)


            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_success_num += path['success']
            n_trajs += 1
            if infer==True:
                self.policy.infer_posterior(self.policy.context)
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                policy.sample_z()
        if n_trajs >= 5:
            success=n_success_num/n_trajs
        return paths, n_steps_total,dict(n_success_num=n_success_num, n_trajs=n_trajs, success=success)

