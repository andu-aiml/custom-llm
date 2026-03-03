import torch
from torch.utils.data import Dataset, DataLoader

class TokenStreamDataset(Dataset):
    """
    Takes a long 1D tensor of token ids and returns (x,y) where:
      x = tokens[i : i+T]
      y = tokens[i+1 : i+T+1]
    """
    def __init__(self, token_ids_1d: torch.Tensor, context_len: int):
        assert token_ids_1d.dim() == 1
        self.tokens = token_ids_1d
        self.T = context_len

        # number of full windows
        self.n = (len(self.tokens) - 1) // self.T
        if self.n <= 0:
            raise ValueError("Not enough tokens for the given context_len")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.T
        x = self.tokens[start : start + self.T]
        y = self.tokens[start + 1 : start + self.T + 1]
        return x, y


def encode_texts_to_1d(tok, texts, add_special=True):
    """
    Tokenize list[str] -> one long 1D torch tensor.
    If you trained with TemplateProcessing [BOS] $A [EOS], you can keep add_special=True.
    """
    all_ids = []
    for t in texts:
        if not t or not t.strip():
            continue
        enc = tok.encode(t)
        ids = enc.ids
        all_ids.extend(ids)

    return torch.tensor(all_ids, dtype=torch.long)


def make_loaders(tok, train_texts, valid_texts, context_len, batch_size=16, num_workers=0):
    train_ids = encode_texts_to_1d(tok, train_texts)
    valid_ids = encode_texts_to_1d(tok, valid_texts)

    train_ds = TokenStreamDataset(train_ids, context_len)
    valid_ds = TokenStreamDataset(valid_ids, context_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, drop_last=False
    )
    return train_loader, valid_loader