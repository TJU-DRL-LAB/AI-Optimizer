import numpy as np
from collections import deque
import random
random.seed(0)

import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)


class Buffer(object):
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.agent_count = args.agent_count
        self.experience_count = 0
        self.buffer = deque()
        self.agent_specific_key_list = None
        self.shared_key_list = None

    def append(self, experience_dict):
        if self.experience_count == 0:
            self.agent_specific_key_list = experience_dict["agent_specific"].keys()
            self.shared_key_list = experience_dict["shared"].keys()

        if self.experience_count < self.buffer_size:
            self.buffer.append(experience_dict)
            self.experience_count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience_dict)

    def sample(self):
        if self.experience_count < self.batch_size:
            batch_experience = random.sample(self.buffer, self.experience_count)
            sampled_count = self.experience_count
        else:
            batch_experience = random.sample(self.buffer, self.batch_size)
            sampled_count = self.batch_size

        experience_dict = {"agent_specific": {}, "shared": {}}
        for key in self.agent_specific_key_list:
            experience_dict["agent_specific"][key] = []
            for i in range(self.agent_count):
                batch = np.asarray([e["agent_specific"][key][i] for e in batch_experience]).reshape(sampled_count, -1)
                experience_dict["agent_specific"][key].append(to_tensor(batch))
        for key in self.shared_key_list:
            batch = np.asarray([e["shared"][key] for e in batch_experience]).reshape(sampled_count, -1)
            experience_dict["shared"][key] = to_tensor(batch)
        return experience_dict

