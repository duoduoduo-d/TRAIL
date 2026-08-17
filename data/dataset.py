from torch.utils.data import Dataset, Subset


class TranscriptDataset(Dataset):

    def __init__(
        self,
        X,
        Y,
        tx_ids
    ):
        self.X = X
        self.Y = Y
        self.tx_ids = tx_ids

        self.tx_to_idx = {
            tx: i
            for i, tx in enumerate(tx_ids)
        }


    def __len__(self):
        return len(self.tx_ids)


    def __getitem__(self, idx):

        return {
            "x": self.X[idx],
            "y": self.Y[idx],
            "tx_id": self.tx_ids[idx]
        }


    def subset(self, tx_ids):

        indices = [
            self.tx_to_idx[x]
            for x in tx_ids
        ]

        return Subset(
            self,
            indices
        )