import torch

class Dataloader(object):
    def __init__(self, data, batch_size=64, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.size = data.size(0)
        self.data = data
        self.iter = 0

    def _sampler(self, size, batch_size, shuffle=True):
        if shuffle:
            idxs = torch.randperm(size)
        else:
            idxs = torch.arange(size)
        for batch_idxs in idxs.split(batch_size):
            yield batch_idxs

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == 0:
            self.sampler = self._sampler(self.size, self.batch_size, shuffle=self.shuffle)
        self.iter = (self.iter + 1) % (len(self)+1)
        idx = next(self.sampler)
        return self.data[idx]

    def __len__(self):
        return (self.size-1)//self.batch_size+1

def generate_dataset_1x2(n_agents, n_items, num_examples, item1_range=(0,1), item2_range=(0,1)):
    item1_min, item1_max = item1_range
    item1_vs = (item1_max - item1_min)*torch.rand((num_examples, n_agents, 1)) + item1_min
    item2_min, item2_max = item2_range
    item2_vs = (item2_max - item2_min)*torch.rand((num_examples, n_agents, 1)) + item2_min
    return torch.cat( (item1_vs, item2_vs), dim=2)

def generate_dataset_nxk(n_agents, n_items, num_examples, item_ranges=None):
    if item_ranges is None:
        item_ranges = [(0, 1) for _ in range(n_items)]
    assert len(item_ranges) == n_items

    all_item_dists = []
    for item in range(n_items):
        item_min, item_max = item_ranges[item]
        all_item_dists.append((item_max - item_min)*torch.rand((num_examples, n_agents, 1)) + item_min)
    return torch.cat(all_item_dists, dim=2)
