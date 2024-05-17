from torch.utils.data import Subset


# Create a class that extends Subset and includes the custom methods
class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def collate(self, batch):
        return self.dataset.collate(batch)
