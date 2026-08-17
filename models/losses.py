import torch
import torch.nn.functional as F


def masked_bce_loss(
    logits,
    targets,
    seq_mask,
    pos_weight=5.0
):
    """
    Masked BCE loss for base-resolution prediction.

    Parameters
    ----------
    logits:
        (B, R, L)

    targets:
        (B, R, L)

    seq_mask:
        (B, L)

    pos_weight:
        positive class weight

    Returns
    -------
    scalar loss
    """


    mask = (
        seq_mask
        .unsqueeze(1)
        .expand_as(logits)
        .to(dtype=logits.dtype)
    )


    if torch.is_tensor(pos_weight):

        weight = pos_weight.to(
            device=logits.device,
            dtype=logits.dtype
        )

    else:

        weight = torch.tensor(
            pos_weight,
            device=logits.device,
            dtype=logits.dtype
        )


    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(
            dtype=logits.dtype
        ),
        reduction="none",
        pos_weight=weight
    )


    loss = loss * mask


    return (
        loss.sum()
        /
        (mask.sum() + 1e-8)
    )




def masked_focal_loss(
    logits,
    targets,
    seq_mask,
    alpha=0.75,
    gamma=2.0
):
    """
    Focal loss for sparse occupancy prediction.
    """


    mask = (
        seq_mask
        .unsqueeze(1)
        .expand_as(logits)
        .float()
    )


    prob = torch.sigmoid(
        logits
    )


    targets = targets.float()


    pt = (
        prob * targets
        +
        (1-prob)*(1-targets)
    )


    loss = (
        -alpha
        *
        (1-pt).pow(gamma)
        *
        torch.log(pt + 1e-8)
    )


    loss = loss * mask


    return (
        loss.sum()
        /
        (mask.sum()+1e-8)
    )