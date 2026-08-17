import os
import sys

import torch
import torch.nn as nn


# ======================================================
# Add local hydra implementation
# Project structure:
#
# RNA_model/
# ├── hydra-main/
# │   └── hydra/
# │
# └── models/
#     └── mamba_block.py
#
# ======================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


HYDRA_PATH = os.path.join(
    PROJECT_ROOT,
    "hydra-main"
)


if HYDRA_PATH not in sys.path:
    sys.path.insert(
        0,
        HYDRA_PATH
    )


from hydra.modules.hydra import Hydra



class Residual(nn.Module):
    """
    Residual connection wrapper.
    """

    def __init__(
        self,
        fn
    ):
        super().__init__()

        self.fn = fn


    def forward(
        self,
        x
    ):

        return self.fn(x) + x




class BiMambaBlock(nn.Module):
    """
    Bidirectional state-space modeling block.

    Input
    -----
    x:
        (batch, length, hidden_dim)


    Output
    ------
    x:
        (batch, length, hidden_dim)

    """


    def __init__(
        self,
        d_model=640,
        d_state=64,
        d_conv=7,
        expand=2,
        headdim=80
    ):

        super().__init__()



        self.norm = nn.LayerNorm(
            d_model
        )



        self.mamba = Hydra(

            d_model=d_model,

            d_state=d_state,

            d_conv=d_conv,

            expand=expand,

            headdim=headdim,

            use_mem_eff_path=False

        )



        self.projection = nn.Linear(
            d_model,
            d_model
        )




    def forward(
        self,
        x
    ):


        residual = x



        x = self.norm(
            x
        )


        x = self.mamba(
            x
        )


        x = self.projection(
            x
        )



        return x