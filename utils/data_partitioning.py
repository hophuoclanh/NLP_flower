import numpy as np

def partition_data(data, num_partitions):
    """Partition data into `num_partitions` for federated clients."""
    data_split = np.array_split(data, num_partitions)
    return {i: data_split[i] for i in range(num_partitions)}

def create_client_datasets(client_data, labels):
    """Create datasets for each client."""
    from torch.utils.data import DataLoader, Dataset

    class NewsDataset(Dataset):
        def __init__(self, tokens, labels):
            self.tokens = tokens
            self.labels = labels

        def __len__(self):
            return len(self.tokens)

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.tokens.items()}, self.labels[idx]

    client_datasets = {
        client_id: DataLoader(
            NewsDataset(client_data[client_id], labels[client_id]), batch_size=32
        )
        for client_id in client_data
    }
    return client_datasets
