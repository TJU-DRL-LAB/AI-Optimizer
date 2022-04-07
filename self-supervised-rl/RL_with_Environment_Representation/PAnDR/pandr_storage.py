from collections import namedtuple
import random


# TransitionPDVF = namedtuple('Transition',
#                         ('state', 'emb_policy', 'emb_env', 'total_return'))

TransitionPDVF = namedtuple('Transition',
                        ('state', 'emb_policy', 'emb_env', 'total_return'))
TransitionPDVF_vl = namedtuple('Transition',
                        ('state', 'policy_data', 'policy_mask', 'env_data', 'env_mask', 'total_return'))
TransitionPolicyDecoder = namedtuple('TransitionPolicyDecoder',
                        ('emb_state', 'recurrent_state', 'mask', 'action'))

class ReplayMemoryPDVF_vl(object):
    '''
    Storage for training the PDVF.
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = TransitionPDVF_vl(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        

class ReplayMemoryPDVF(object):
    '''
    Storage for training the PDVF. 
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = TransitionPDVF(*args)            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemoryPolicyDecoder(object):
    '''
    Storage for training the olicy decoder. 
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = TransitionPolicyDecoder(*args)            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)