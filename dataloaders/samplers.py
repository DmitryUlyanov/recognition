import numpy as np
from torch.utils.data.sampler import Sampler

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, num_batches, batch_size, proportions=None, replacement=False):
        # super().__init__(data_source)
        self.labels = labels
        self.replacement = replacement
        self.num_batches = num_batches
        self.batch_size = batch_size


        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)

        # fractions = [0.95, 0.05]
        if proportions is None:
            self.class_per_batch = {cls: int(self.batch_size / self.num_classes) for cls in self.classes}
        else:
            self.class_per_batch = {cls: int(self.batch_size * proportions[i]) for i, cls in enumerate(self.classes)}


        print('P', self.class_per_batch, self.num_classes, self.batch_size, self.replacement)

        self.class2indices = {cls: np.where(self.labels == cls)[0] for cls in self.classes}
        

    def __iter__(self):
        for i in range(self.num_batches):
            batch = []

            for cls in self.class2indices:
                batch.extend(np.random.choice(self.class2indices[cls], self.class_per_batch[cls], replace=self.replacement))
    
            
            if len(batch) < self.batch_size:
                batch = list(batch) + list(np.random.choice(np.arange(len(self.labels)), self.batch_size - len(batch)))
            
            batch = np.random.permutation(batch)

            yield batch

    def __len__(self):
        return self.num_batches