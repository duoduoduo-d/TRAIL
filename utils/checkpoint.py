import os
import torch




def save_checkpoint(
    path,
    model,
    config=None,
    extra_info=None
):
    """
    Save model checkpoint.

    Designed for model release.

    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )



    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "config":
            config,

    }



    if extra_info is not None:

        checkpoint.update(
            extra_info
        )



    torch.save(
        checkpoint,
        path
    )





def load_checkpoint(
    path,
    model,
    device="cpu"
):
    """
    Load model parameters.
    """

    checkpoint = torch.load(
        path,
        map_location=device
    )



    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint["model_state_dict"]
        )

    else:

        # support raw state dict

        state_dict = checkpoint



    model.load_state_dict(
        state_dict
    )


    return checkpoint