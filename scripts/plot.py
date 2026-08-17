import os
import sys
import argparse

import torch
import pandas as pd


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

    targets = ckpt["targets"]


    model = TRAIL(

        input_channels=
        config["model"]["input_channels"],

        hidden_dim=
        config["model"]["hidden_dim"],

        depth=
        config["model"]["depth"],

        num_targets=
        len(targets),

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


    return model, targets, config



# ======================================================
# Get transcript
# ======================================================

def get_transcript(
    transcript_file,
    tx_id
):

    df = pd.read_csv(
        transcript_file,
        sep="\t"
    )


    row = df[
        df["tx_id"] == tx_id
    ]


    if len(row)==0:

        raise ValueError(
            f"{tx_id} not found"
        )


    return row.iloc[0]



# ======================================================
# Build input
# ======================================================

def build_input(
    row,
    config
):

    x, offset, end = encode_transcript(

        sequence=row["sequence"],

        gene_type=row["gene_type"],

        cds_start=row["cds_start"],

        cds_end=row["cds_end"],

        tx_size=row["tx_size"],

        splice_sites=row["splice"],

        max_length=
        config["data"]["max_length"],

        max_utr5_len=
        config["data"]["max_utr5_len"]

    )


    return x.unsqueeze(0)



# ======================================================
# Predict
# ======================================================

@torch.no_grad()
def predict(
    model,
    x,
    device
):

    x = x.to(device)


    logits, mask = model(x)


    prob = torch.sigmoid(
        logits
    )


    # [1,target,length]

    prob = prob[0].cpu()

    mask = mask[0].cpu().bool()


    # remove padding

    prob = prob[:, mask]


    return prob



# ======================================================
# Save
# ======================================================

def save_prediction(
    prob,
    targets,
    output
):


    data = {}


    for i,t in enumerate(targets):

        data[t] = (
            prob[i]
            .numpy()
        )


    df = pd.DataFrame(
        data
    )


    df.to_csv(
        output,
        sep="\t",
        index=False
    )

def plot_prediction(
    prob,
    targets,
    tx_row,
    tx_id,
    output_dir="predict_results",
    max_per_fig=10
):

    import os
    import numpy as np
    import matplotlib.pyplot as plt


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # ==========================
    # basic information
    # ==========================

    length = prob.shape[1]

    x = np.arange(length)


    gene_name = str(
        tx_row["gene_name"]
    )

    gene_type = str(
        tx_row["gene_type"]
    )



    # ==========================
    # region annotation
    # ==========================

    cds_start = None
    cds_end = None


    if gene_type == "protein_coding":

        cds_start = int(
            tx_row["cds_start"]
        )

        cds_end = int(
            tx_row["cds_end"]
        )


        cds_start = max(
            0,
            min(cds_start, length)
        )

        cds_end = max(
            0,
            min(cds_end, length)
        )



    # ==========================
    # colors
    # ==========================

    color_list = [

        "#5EA0C7",
        "#6BC179",
        "#F2A65A",
        "#C998C9",
        "#E07A7A",
        "#7BC8C4",
        "#A6BCE0",
        "#B4D7E5",
        "#D9A6A6",
        "#9CCB86"

    ]


    color_map = {

        t:
        color_list[i % len(color_list)]

        for i,t in enumerate(targets)

    }


    region_colors = {

        "5UTR": "#5EA0C7",

        "CDS": "#6BC179",

        "3UTR": "#9FD4C5"

    }


    divider_color = "#AFC6D6"



    # ==========================
    # plotting
    # ==========================

    for start in range(
        0,
        len(targets),
        max_per_fig
    ):


        end = min(
            start + max_per_fig,
            len(targets)
        )


        rbp_list = targets[start:end]



        # dynamic width

        fig_width = np.clip(
            10 + length / 3000,
            10,
            15
        )


        fig, ax = plt.subplots(
            figsize=(
                fig_width,
                4
            )
        )



        # --------------------------
        # prediction curves
        # --------------------------

        for i,rbp in enumerate(rbp_list):


            idx = start+i


            y = prob[idx]


            color = color_map[rbp]


            ax.plot(

                x,

                y,

                lw=1.8,

                color=color,

                alpha=0.95,

                label=rbp.split("_")[-1]

            )


            ax.fill_between(

                x,

                0,

                y,

                color=color,

                alpha=0.05,

                linewidth=0

            )



        # --------------------------
        # region annotation
        # --------------------------

        if gene_type == "protein_coding":


            ax.axvline(

                cds_start,

                linestyle="--",

                linewidth=1.2,

                color=divider_color,

                alpha=0.8

            )


            ax.axvline(

                cds_end,

                linestyle="--",

                linewidth=1.2,

                color=divider_color,

                alpha=0.8

            )


            label_y = 1.01



            if cds_start > 5:

                ax.text(

                    cds_start/2,

                    label_y,

                    "5′UTR",

                    ha="center",

                    fontsize=10,

                    fontweight="bold",

                    color=region_colors["5UTR"]

                )



            ax.text(

                (cds_start+cds_end)/2,

                label_y,

                "CDS",

                ha="center",

                fontsize=10,

                fontweight="bold",

                color=region_colors["CDS"]

            )



            if cds_end < length-5:

                ax.text(

                    (cds_end+length)/2,

                    label_y,

                    "3′UTR",

                    ha="center",

                    fontsize=10,

                    fontweight="bold",

                    color=region_colors["3UTR"]

                )


        else:


            ax.text(

                length/2,

                1.01,

                "lncRNA",

                ha="center",

                fontsize=10,

                color="#777777"

            )



        # --------------------------
        # style
        # --------------------------

        ax.set_ylim(
            -0.02,
            1.05
        )


        ax.set_xlim(
            0,
            length
        )


        ax.set_ylabel(

            "Predicted score",

            fontsize=11,

            fontname="Arial"

        )


        ax.set_xlabel(

            "Transcript position",

            fontsize=11,

            fontname="Arial"

        )


        ax.set_title(

            f"{gene_name} | {tx_id} ({gene_type})",

            fontsize=13,

            pad=10,

            weight="bold",

            fontname="Arial"

        )


        ax.legend(

            frameon=False,

            ncol=min(len(rbp_list),5),

            loc="upper right",

            prop={"family":"Arial"}

        )


        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)


        ax.grid(

            axis="y",

            linestyle="--",

            linewidth=0.6,

            alpha=0.25

        )


        plt.tight_layout()



        save_path = os.path.join(

            output_dir,

            f"{tx_id}_targets_{start//max_per_fig+1}.pdf"

        )


        plt.savefig(

            save_path,

            dpi=350,

            bbox_inches="tight"

        )


        #plt.show()

        plt.close()

# ======================================================
# Main
# ======================================================

def main(args):


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model, all_targets, config = load_model(
        args.checkpoint,
        device
    )



    # ----------------------
    # target selection
    # ----------------------

    if args.targets == ["all"]:

        selected_targets = all_targets


    else:

        selected_targets = args.targets



    indices = []


    for t in selected_targets:


        if t not in all_targets:

            raise ValueError(
                f"{t} not found in checkpoint"
            )


        indices.append(
            all_targets.index(t)
        )



    # ----------------------
    # transcript
    # ----------------------

    row = get_transcript(

        args.transcript_file,

        args.tx_id

    )


    x = build_input(
        row,
        config
    )



    # ----------------------
    # inference
    # ----------------------

    prob = predict(

        model,

        x,

        device

    )


    prob = prob[
        indices
    ]



    selected_targets = [

        all_targets[i]

        for i in indices

    ]



    # ----------------------
    # save
    # ----------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    output = os.path.join(

        args.output_dir,

        f"{args.tx_id}_prediction.tsv"

    )


    save_prediction(

        prob,

        selected_targets,

        output

    )
    
    plot_prediction(
        prob=prob,
        targets=selected_targets,
        tx_row=row,
        tx_id=args.tx_id,
        output_dir=args.output_dir
    )


    print(
        "Saved:",
        output
    )


    print(
        "Shape:",
        prob.shape
    )



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
        "--tx_id",
        required=True
    )


    parser.add_argument(
        "--targets",
        nargs="+",
        required=True
    )


    parser.add_argument(
        "--output_dir",
        default="predict_results"
    )


    args = parser.parse_args()


    main(args)