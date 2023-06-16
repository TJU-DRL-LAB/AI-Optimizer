import torch
import numpy as np
from tqdm import tqdm


class DistNetworkTrainer(object):
    def __init__(self, dist_model, learning_rate=1e-3):
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self._dist_model = dist_model.cuda(self.device)
        self._opt = torch.optim.Adam(dist_model.parameters(), lr=learning_rate)

    def train(self, train_data):
        for i in tqdm(range(len(train_data))):
            state_offline = torch.from_numpy(train_data[i]).cuda(self.device)
            pred_dist = self._dist_model(state_offline)

            for j in range(20):
                state_sample_1 = torch.rand(1, 1) * 8
                state_sample_2 = torch.rand(1, 1) * 11
                state_sample_3 = torch.rand(1, 1) * 12 - 6
                state_sample_4 = torch.rand(1, 1) * 12 - 6
                state_sample = torch.cat([state_sample_1, state_sample_2], dim=1)
                state_sample = torch.cat([state_sample, state_sample_3], dim=1)
                state_sample = torch.cat([state_sample, state_sample_4], dim=1)
                state_sample = state_sample.squeeze().cuda(self.device)

                dist = torch.norm(state_offline - state_sample).cuda(self.device)
                loss = torch.nn.functional.binary_cross_entropy(pred_dist, dist)

                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
