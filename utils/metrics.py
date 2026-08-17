import torch

from torchmetrics.classification import (
    AUROC,
    AveragePrecision
)


class TargetMetrics:
    """
    Metrics for multi-target nucleotide-resolution prediction.

    predictions:
        (B, R, L)

    targets:
        (B, R, L)

    Calculate:

        1. AUROC/AUPR for each target

        2. Mean AUROC/AUPR across targets

    """


    def __init__(
        self,
        num_targets,
        target_names=None,
        device="cpu"
    ):

        self.num_targets = num_targets


        if target_names is None:

            self.target_names = [
                str(i)
                for i in range(num_targets)
            ]

        else:

            self.target_names = target_names



        self.auroc = []

        self.aupr = []


        for _ in range(num_targets):

            self.auroc.append(
                AUROC(
                    task="binary"
                ).to(device)
            )


            self.aupr.append(
                AveragePrecision(
                    task="binary"
                ).to(device)
            )



    @torch.no_grad()
    def update(
        self,
        predictions,
        targets,
        mask=None
    ):
        """
        Update metrics with one batch.

        predictions:
            sigmoid output
            (B,R,L)

        targets:
            binary labels
            (B,R,L)

        mask:
            (B,L)
            valid transcript positions
        """


        batch_size, num_targets, length = (
            predictions.shape
        )


        if mask is not None:

            mask_expand = (
                mask
                .unsqueeze(1)
                .expand(
                    -1,
                    num_targets,
                    -1
                )
                .bool()
            )

        else:

            mask_expand = torch.ones_like(
                targets,
                dtype=torch.bool
            )



        for i in range(
            num_targets
        ):


            pred_i = predictions[
                :,
                i,
                :
            ]


            target_i = targets[
                :,
                i,
                :
            ]



            mask_i = mask_expand[
                :,
                i,
                :
            ]



            pred_i = pred_i[
                mask_i
            ]


            target_i = target_i[
                mask_i
            ]



            if target_i.numel() == 0:

                continue



            # torchmetrics requires int/long target

            target_i = target_i.long()



            self.auroc[i].update(
                pred_i.float(),
                target_i
            )


            self.aupr[i].update(
                pred_i.float(),
                target_i
            )



    def compute(self):
        """
        Return mean metrics and per-target metrics.
        """


        results = {

            "overall":
            {},

            "per_target":
            {}

        }


        auroc_values = []

        aupr_values = []



        for i, name in enumerate(
            self.target_names
        ):


            auroc = (
                self.auroc[i]
                .compute()
                .item()
            )


            aupr = (
                self.aupr[i]
                .compute()
                .item()
            )


            auroc_values.append(
                auroc
            )

            aupr_values.append(
                aupr
            )


            results["per_target"][name] = {

                "auroc":
                auroc,

                "aupr":
                aupr

            }



        if len(auroc_values) > 0:

            results["overall"] = {

                "auroc":
                sum(auroc_values)
                /
                len(auroc_values),


                "aupr":
                sum(aupr_values)
                /
                len(aupr_values)

            }

        else:

            results["overall"] = {

                "auroc": 0.0,

                "aupr": 0.0

            }



        return results



    def reset(self):
        """
        Reset metrics.
        """


        for metric in self.auroc:

            metric.reset()



        for metric in self.aupr:

            metric.reset()