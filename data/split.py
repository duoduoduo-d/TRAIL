import os
import json
import random


def create_split(tx_ids, split_file, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if os.path.exists(split_file):
        with open(split_file) as f:
            return json.load(f)

    ids = list(tx_ids)
    random.seed(seed)
    random.shuffle(ids)

    n = len(ids)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    split = {
        "train": ids[val_size + test_size:],
        "valid": ids[test_size:test_size + val_size],
        "test": ids[:test_size]
    }

    os.makedirs(
        os.path.dirname(split_file),
        exist_ok=True
    )

    with open(split_file, "w") as f:
        json.dump(split, f)

    print(
        f"Split: {len(split['train'])} train, "
        f"{len(split['valid'])} valid, "
        f"{len(split['test'])} test"
    )

    return split