import os
import sys
import argparse

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from data.features import encode_transcript
from models.model import TRAIL



# ======================================================
# Load model
# ======================================================

def load_model(
    checkpoint_path,
    device
):

    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )


    config = ckpt["config"]

    all_targets = ckpt["targets"]


    model = TRAIL(

        input_channels=
        config["model"]["input_channels"],

        hidden_dim=
        config["model"]["hidden_dim"],

        depth=
        config["model"]["depth"],

        num_targets=
        len(all_targets),

        dropout=
        config["model"].get(
            "dropout",
            0.1
        )

    )


    model.load_state_dict(
        ckpt["model_state_dict"]
    )


    model.to(device)

    model.eval()


    return model, all_targets, config



# ======================================================
# Dataset
# ======================================================

class TranscriptDataset(Dataset):


    def __init__(
        self,
        transcript_file,
        config
    ):

        self.df = pd.read_csv(
            transcript_file,
            sep="\t"
        )

        self.config = config



    def __len__(self):

        return len(self.df)



    def __getitem__(
        self,
        idx
    ):


        row = self.df.iloc[idx]


        x, offset, end = encode_transcript(

            sequence=row["sequence"],

            gene_type=row["gene_type"],

            cds_start=row["cds_start"],

            cds_end=row["cds_end"],

            tx_size=row["tx_size"],

            splice_sites=row["splice"],

            max_length=
            self.config["data"]["max_length"],

            max_utr5_len=
            self.config["data"]["max_utr5_len"]

        )


        return {

            "x":
            x,

            "tx_id":
            row["tx_id"],

            "offset":
            offset,

            "end":
            end

        }



# ======================================================
# Collate
# ======================================================

def collate_fn(batch):


    x = torch.stack(
        [
            item["x"]
            for item in batch
        ]
    )


    return {


        "x":
        x,


        "tx_ids":
        [
            item["tx_id"]
            for item in batch
        ],


        "offsets":
        [
            item["offset"]
            for item in batch
        ],


        "ends":
        [
            item["end"]
            for item in batch
        ]

    }



# ======================================================
# Predict batch
# ======================================================

@torch.no_grad()
def predict_batch(
    model,
    x,
    device
):


    x = x.to(
        device,
        non_blocking=True
    )


    logits, mask = model(x)



    # logits:
    #
    # [B,target,length]


    prob = torch.sigmoid(
        logits
    )


    # remove padding
    #
    # mask:
    # [B,length]
    #

    prob = prob * mask.unsqueeze(1)



    # 
    # [B,target,length]
    #
    # ->
    #
    # [B,length,target]
    #

    prob = prob.permute(
        0,
        2,
        1
    )


    return prob



# ======================================================
# Main
# ======================================================

def main(args):


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        "Device:",
        device
    )



    model, all_targets, config = load_model(

        args.checkpoint,

        device

    )



    # ==================================================
    # target selection
    # ==================================================

    if args.targets == ["all"]:


        selected_targets = all_targets


        target_indices = list(
            range(
                len(all_targets)
            )
        )


    else:


        selected_targets = args.targets


        target_indices = []


        for t in selected_targets:


            if t not in all_targets:

                raise ValueError(
                    f"{t} not found in checkpoint"
                )


            target_indices.append(
                all_targets.index(t)
            )



    print(
        "Selected targets:",
        len(selected_targets)
    )



    # ==================================================
    # Dataset
    # ==================================================

    dataset = TranscriptDataset(

        args.transcript_file,

        config

    )



    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=True,

        collate_fn=collate_fn

    )



    all_predictions = []

    all_offsets = []

    all_ends = []

    all_tx_ids = []



    # ==================================================
    # inference
    # ==================================================

    for batch in tqdm(

        loader,

        desc="Predicting"

    ):


        prob = predict_batch(

            model,

            batch["x"],

            device

        )


        #
        # [B,L,target]
        #

        prob = prob[
            :,
            :,
            target_indices
        ]



        # save CPU float16

        prob = prob.cpu().half()



        all_predictions.append(
            prob
        )


        all_offsets.extend(
            batch["offsets"]
        )


        all_ends.extend(
            batch["ends"]
        )


        all_tx_ids.extend(
            batch["tx_ids"]
        )



    # ==================================================
    # concatenate
    # ==================================================

    predictions = torch.cat(

        all_predictions,

        dim=0

    )



    output = {


        "predictions":

        predictions,


        "offsets":

        torch.tensor(

            all_offsets,

            dtype=torch.long

        ),


        "ends":

        torch.tensor(

            all_ends,

            dtype=torch.long

        ),


        "tx_ids":

        all_tx_ids,


        "targets":

        selected_targets

    }



    torch.save(

        output,

        args.output

    )



    print(
        "Saved:",
        args.output
    )


    print(
        "Prediction shape:",
        predictions.shape
    )



# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":


    parser = argparse.ArgumentParser()



    parser.add_argument(
        "--checkpoint",
        required=True
    )


    parser.add_argument(
        "--transcript_file",
        required=True
    )


    parser.add_argument(
        "--output",
        required=True
    )


    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"]
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )


    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )


    args = parser.parse_args()


    main(args)