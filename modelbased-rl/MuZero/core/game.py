from typing import List

import numpy as np
import torch


class Player(object):
    def __init__(self, id=1):
        self.id = id

    def __eq__(self, other):
        if not isinstance(other, Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id


class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()


class Game:
    def __init__(self, env, action_space_size: int, discount: float, config=None):
        self.env = env
        self.obs_history = []
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.config = config

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, model=None, config=None):
        # The value target is the discounted root value of the search tree N steps into the future, plus
        # the discounted sum of all rewards until then.
        target_values, target_rewards, target_policies = [], [], []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                if model is None:
                    value = self.root_values[bootstrap_index] * self.discount ** td_steps
                else:
                    # Reference : Appendix H => Reanalyze
                    # Note : a target network  based on recent parameters is used to provide a fresher,
                    # stable n-step bootstrapped target for the value function
                    obs = self.obs(bootstrap_index)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_inference(obs)
                    value = network_output.value.data.cpu().item() * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index-1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                target_values.append(value)
                target_rewards.append(last_reward)

                # Reference : Appendix H => Reanalyze
                # Note : MuZero Reanalyze revisits its past time-steps and re-executes its search using the
                # latest model parameters, potentially resulting in a better quality policy than the original search.
                # This fresh policy is used as the policy target for 80% of updates during MuZero training
                if model is not None and np.random.random() <= config.revisit_policy_search_rate:
                    from core.mcts import MCTS, Node
                    root = Node(0)
                    obs = self.obs(current_index)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_inference(obs)
                    root.expand(self.to_play(), self.legal_actions(), network_output)
                    MCTS(config).run(root, self.action_history(current_index), model)
                    self.store_search_stats(root, current_index)

                target_policies.append(self.child_visits[current_index])

            else:
                # States past the end of games are treated as absorbing states.
                target_values.append(0)
                target_rewards.append(last_reward)
                # Note: Target policy is  set to 0 so that no policy loss is calculated for them
                target_policies.append([0 for _ in range(len(self.child_visits[0]))])

        return target_values, target_rewards, target_policies

    def action_history(self, idx=None) -> ActionHistory:
        if idx is None:
            return ActionHistory(self.history, self.action_space_size)
        else:
            return ActionHistory(self.history[:idx], self.action_space_size)

    def store_search_stats(self, root, idx: int = None):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        if idx is None:
            self.child_visits.append([root.children[a].visit_count / sum_visits if a in root.children else 0
                                      for a in action_space])
            self.root_values.append(root.value())
        else:
            self.child_visits[idx] = [root.children[a].visit_count / sum_visits if a in root.children else 0
                                      for a in action_space]
            self.root_values[idx] = root.value()

    def to_play(self) -> Player:
        return Player()

    def __len__(self):
        return len(self.rewards)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)
