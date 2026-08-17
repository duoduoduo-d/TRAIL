import os
import sys
import yaml
import json
import random
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm


from data.preprocess import TranscriptDataModule
from data.split import create_split

from models.model import TRAIL
from models.losses import masked_bce_loss

from utils.metrics import TargetMetrics

import random
import string

exp_id = ''.join(
    random.choices(
        string.ascii_lowercase + string.digits,
        k=6
    )
)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)



def load_config(path):

    with open(path) as f:

        return yaml.safe_load(f)



def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    pos_weight
):

    model.train()

    total_loss = 0


    pbar = tqdm(
        loader,
        desc="Training",
        leave=False
    )


    for batch in pbar:


        x = batch["x"].to(
            device,
            non_blocking=True
        )

        y = batch["y"].to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad()


        with autocast(
            device_type=device.type,
            enabled=device.type=="cuda"
        ):


            logits, seq_mask = model(x)


            loss = masked_bce_loss(
                logits,
                y,
                seq_mask,
                pos_weight
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        total_loss += loss.item()


        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    return total_loss / len(loader)




@torch.no_grad()
def validate(
    model,
    loader,
    device,
    pos_weight,
    num_targets,
    target_names
):

    model.eval()


    total_loss = 0


    metrics = TargetMetrics(
        num_targets=num_targets,
        target_names=target_names,
        device=device
    )


    pbar = tqdm(
        loader,
        desc="Validation",
        leave=False
    )


    for batch in pbar:


        x = batch["x"].to(
            device,
            non_blocking=True
        )


        y = batch["y"].to(
            device,
            non_blocking=True
        )


        logits, seq_mask = model(x)


        loss = masked_bce_loss(
            logits,
            y,
            seq_mask,
            pos_weight
        )


        total_loss += loss.item()


        prob = torch.sigmoid(
            logits
        ).float()


        metrics.update(
            prob,
            y,
            seq_mask
        )


    result = metrics.compute()


    return (
        total_loss / len(loader),
        result
    )




def main(config_path):


    config = load_config(
        config_path
    )


    set_seed(
        config.get(
            "seed",
            42
        )
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ==========================
    # Dataset
    # ==========================

    dm = TranscriptDataModule(

        transcript_file=
        config["data"]["transcript_file"],

        label_dir=
        config["data"]["label_dir"],

        cache_dir=
        config["data"]["cache_dir"],

        targets=
        config["targets"],

        cell_line=
        config["cell_line"],

        tpm_threshold=
        config["data"]["tpm_threshold"],

        max_length=
        config["data"]["max_length"],

        max_utr5_len=
        config["data"]["max_utr5_len"]

    )


    dm.setup()



    # ==========================
    # Split
    # ==========================

    split = create_split(

        dm.tx_ids,

        config["data"]["split_file"],

        seed=config.get(
            "seed",
            42
        )

    )


    train_dataset = dm.dataset.subset(
        split["train"]
    )


    valid_dataset = dm.dataset.subset(
        split["valid"]
    )


    test_dataset = dm.dataset.subset(
        split["test"]
    )



    batch_size = config["training"]["batch_size"]



    loader_args = {

        "batch_size": batch_size,

        "num_workers": 4,

        "pin_memory": True,

        "persistent_workers": True

    }



    train_loader = DataLoader(

        train_dataset,

        shuffle=True,

        drop_last=True,

        **loader_args

    )


    valid_loader = DataLoader(

        valid_dataset,

        shuffle=False,

        **loader_args

    )


    test_loader = DataLoader(

        test_dataset,

        shuffle=False,

        **loader_args

    )



    # ==========================
    # Model
    # ==========================

    model = TRAIL(

        input_channels=
        config["model"]["input_channels"],

        hidden_dim=
        config["model"]["hidden_dim"],

        depth=
        config["model"]["depth"],

        num_targets=
        len(config["targets"]),

        dropout=
        config["model"].get(
            "dropout",
            0.1
        )

    )


    model.to(device)



    optimizer = AdamW(

        model.parameters(),

        lr=
        config["training"]["learning_rate"],

        weight_decay=
        config["training"]["weight_decay"]

    )


    scheduler = CosineAnnealingLR(

        optimizer,

        T_max=
        config["training"]["epochs"]

    )



    pos_weight = torch.tensor(

        config["training"].get(
            "pos_weight",
            5.0
        ),

        device=device

    )



    scaler = GradScaler(

        device="cuda",

        enabled=device.type=="cuda"

    )



    # ==========================
    # Training
    # ==========================

    best_aupr = -1

    wait = 0

    patience = config["training"].get(
        "patience",
        10
    )


    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    run_name = f"{config['cell_line']}_multi{len(config['targets'])}_tpm{config['data']['tpm_threshold']}_{exp_id}"

    best_path = os.path.join(

        "checkpoints",

        f"{run_name}_best.pt"

    )



    for epoch in range(

        config["training"]["epochs"]

    ):


        train_loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            scaler,

            device,

            pos_weight

        )



        val_loss, result = validate(

            model,

            valid_loader,

            device,

            pos_weight,

            len(config["targets"]),

            config["targets"]

        )


        scheduler.step()



        val_aupr = result["overall"]["aupr"]



        print(

            f"Epoch {epoch+1}: "

            f"train_loss={train_loss:.4f} "

            f"val_loss={val_loss:.4f} "

            f"AUROC={result['overall']['auroc']:.4f} "

            f"AUPR={val_aupr:.4f}"

        )



        if val_aupr > best_aupr:


            best_aupr = val_aupr

            wait = 0



            torch.save(

                {

                    "model_state_dict":
                    model.state_dict(),

                    "config":
                    config,

                    "targets":
                    config["targets"],

                    "aupr":
                    best_aupr,

                },

                best_path

            )


            print(
                "Saved best model."
            )


        else:


            wait += 1


            if wait >= patience:

                print(
                    "Early stopping."
                )

                break



    # ==========================
    # Test
    # ==========================

    checkpoint = torch.load(

        best_path,

        map_location=device

    )


    model.load_state_dict(

        checkpoint["model_state_dict"]

    )



    test_loss, test_result = validate(

        model,

        test_loader,

        device,

        pos_weight,

        len(config["targets"]),

        config["targets"]

    )


    os.makedirs(
        "checkpoints",
        exist_ok=True
    )


    result_path = os.path.join(

        "checkpoints",

        f"{run_name}_test_metrics.json"

    )


    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            test_result,
            f,
            indent=4
        )



    print("\nFinal test performance:")

    print(
        f"AUROC: {test_result['overall']['auroc']:.4f}"
    )

    print(
        f"AUPR: {test_result['overall']['aupr']:.4f}"
    )

    print(
        f"Saved results: {result_path}"
    )




if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--config",

        required=True

    )


    args = parser.parse_args()


    main(
        args.config
    )