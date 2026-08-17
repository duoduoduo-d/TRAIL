import os
import gc
import torch
import pandas as pd

from .features import encode_transcript
from .dataset import TranscriptDataset


class TranscriptDataModule:

    def __init__(
        self,
        transcript_file,
        label_dir,
        cache_dir,
        targets,
        cell_line="K562",
        tpm_threshold=10,
        max_length=13200,
        max_utr5_len=1500
    ):

        self.transcript_file = transcript_file
        self.label_dir = label_dir
        self.cache_dir = cache_dir

        self.targets = targets
        self.cell_line = cell_line

        self.tpm_threshold = tpm_threshold

        self.max_length = max_length
        self.max_utr5_len = max_utr5_len


        self.seq_cache_path = os.path.join(
            cache_dir,
            f"{cell_line}_main_transcript_info_8channel.pt"
        )


        self.label_cache_dir = os.path.join(
            cache_dir,
            "labels"
        )


    # ======================================================
    # Feature cache
    # ======================================================

    def feature_cache_exists(self):

        return os.path.exists(
            self.seq_cache_path
        )
        
    def filter_transcripts(self, df):
        df["sequence"] = (
            df["sequence"]
            .astype(str)
            .str.upper()
            .str.replace("U","T")
        )


        df = df[
            df["gene_type"].isin(
                [
                    "protein_coding",
                    "lncRNA"
                ]
            )
        ]


        is_lncrna = (
            df["gene_type"]=="lncRNA"
        )


        is_pc = (
            df["gene_type"]=="protein_coding"
        )


        lncrna_good = (
            df["tx_size"]
            <
            self.max_length-self.max_utr5_len
        )


        pc_good = (

            (df["cds_start"] < self.max_utr5_len)

            &

            (df["tx_size"] != df["cds_end"])

            &

            (df["cds_start"] > 0)

            &

            (
                df["tx_size"]
                -
                df["cds_start"]
                <
                self.max_length-self.max_utr5_len
            )

        )


        df = df[
            (is_lncrna & lncrna_good)
            |
            (is_pc & pc_good)
        ].reset_index(drop=True)


        return df


    def build_feature_cache(self):

        print(
            "Building transcript feature cache..."
        )


        df = pd.read_csv(
            self.transcript_file,
            sep="\t"
        )


        df = self.filter_transcripts(df)


        X = []
        offsets = []
        end_pos = []
        tx_ids = []


        for _, row in df.iterrows():

            x, offset, end = encode_transcript(
                sequence=row["sequence"],
                gene_type=row["gene_type"],
                cds_start=row["cds_start"],
                cds_end=row["cds_end"],
                tx_size=row["tx_size"],
                splice_sites=row["splice"],
                max_length=self.max_length,
                max_utr5_len=self.max_utr5_len
            )

            X.append(x)
            offsets.append(offset)
            end_pos.append(end)
            tx_ids.append(row["tx_id"])


        X = torch.stack(X)

        offsets = torch.tensor(
            offsets,
            dtype=torch.int32
        )

        end_pos = torch.tensor(
            end_pos,
            dtype=torch.int32
        )


        tpm_cols = [
            c for c in df.columns
            if c.endswith("_mean_tpm")
        ]


        tpm = {

            c: torch.tensor(
                df[c].values,
                dtype=torch.float32
            )

            for c in tpm_cols

        }


        cache = {

            "X": X,

            "tx_ids": tx_ids,

            "offsets": offsets,

            "end_pos": end_pos,

            "tpm": tpm,

            "cell_line": self.cell_line,

            "max_length": self.max_length,

            "max_utr5_len": self.max_utr5_len

        }


        os.makedirs(
            self.cache_dir,
            exist_ok=True
        )


        torch.save(
            cache,
            self.seq_cache_path
        )


        print(
            f"Feature cache saved: {self.seq_cache_path}"
        )



    def load_feature_cache(self):

        print(
            f"Loading feature cache: {self.seq_cache_path}"
        )


        cache = torch.load(
            self.seq_cache_path,
            map_location="cpu"
        )


        if cache["max_length"] != self.max_length:

            raise ValueError(
                "max_length mismatch with cache"
            )


        if cache["max_utr5_len"] != self.max_utr5_len:

            raise ValueError(
                "max_utr5_len mismatch with cache"
            )


        self.X = cache["X"]

        self.tx_ids = cache["tx_ids"]

        self.offsets = cache["offsets"]

        self.end_pos = cache["end_pos"]

        self.tpm = cache["tpm"]



    # ======================================================
    # Label cache
    # ======================================================

    def label_cache_path(
        self,
        target
    ):

        return os.path.join(
            self.label_cache_dir,
            target + ".pt"
        )


    def label_cache_exists(
        self,
        target
    ):

        return os.path.exists(
            self.label_cache_path(target)
        )



    def build_label_cache(
        self,
        target
    ):

        print(
            f"Building label cache: {target}"
        )


        raw_path = os.path.join(
            self.label_dir,
            target + ".txt"
        )


        rbp_df = pd.read_csv(
            raw_path,
            sep="\t"
        )


        tx_to_index = {

            tx:i

            for i, tx in enumerate(
                self.tx_ids
            )

        }


        Y = torch.zeros(
            (
                len(self.tx_ids),
                self.max_length
            ),
            dtype=torch.float16
        )


        target_col = target



        for _, row in rbp_df.iterrows():

            tx = row["tx_id"]


            if tx not in tx_to_index:
                continue


            idx = tx_to_index[tx]


            offset = self.offsets[idx].item()


            signal = row[target_col]


            for pos, value in enumerate(signal):

                if value == "1":

                    model_pos = offset + pos


                    if (
                        0 <= model_pos <
                        self.max_length
                    ):

                        Y[
                            idx,
                            model_pos
                        ] = 1.0



        os.makedirs(
            self.label_cache_dir,
            exist_ok=True
        )


        torch.save(

            {
                "Y": Y,

                "tx_ids": self.tx_ids,

                "target_name": target

            },

            self.label_cache_path(target)

        )


        return Y



    def load_label_cache(
        self,
        target
    ):

        data = torch.load(
            self.label_cache_path(target),
            map_location="cpu"
        )


        if data["tx_ids"] != self.tx_ids:

            raise ValueError(
                f"{target} cache tx_id mismatch"
            )


        return data["Y"]



    # ======================================================
    # Setup
    # ======================================================

    def setup(self):

        # -------------------------
        # feature
        # -------------------------

        if self.feature_cache_exists():

            self.load_feature_cache()

        else:

            self.build_feature_cache()

            self.load_feature_cache()



        # 保存完整index
        full_tx_ids = self.tx_ids.copy()



        # -------------------------
        # labels
        # -------------------------

        Y_list = []


        for target in self.targets:


            if self.label_cache_exists(target):

                Y = self.load_label_cache(
                    target
                )

            else:

                Y = self.build_label_cache(
                    target
                )


            Y_list.append(
                Y.unsqueeze(1)
            )



        self.Y = torch.cat(
            Y_list,
            dim=1
        )



        # -------------------------
        # TPM filter
        # -------------------------

        tpm_col = (
            f"{self.cell_line}_mean_tpm"
        )


        if tpm_col in self.tpm:

            mask = (
                self.tpm[tpm_col]
                >
                self.tpm_threshold
            )

            indices = torch.where(
                mask
            )[0]

        else:

            indices = torch.arange(
                len(full_tx_ids)
            )



        self.X = self.X[indices]

        self.Y = self.Y[indices]

        self.offsets = self.offsets[indices]

        self.end_pos = self.end_pos[indices]


        self.tx_ids = [
            full_tx_ids[i]
            for i in indices.tolist()
        ]



        self.dataset = TranscriptDataset(
            self.X,
            self.Y,
            self.tx_ids
        )


        gc.collect()


        print(
            "Dataset ready:",
            self.X.shape,
            self.Y.shape
        )