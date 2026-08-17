import torch
import torch.nn as nn

from .mamba_block import BiMambaBlock, Residual


class TRAIL(nn.Module):

    """
    Transcript-scale regulatory landscape model.

    Input:
        x:
            (B, 8, L)

    Output:
        logits:
            (B, targets, L)

        seq_mask:
            (B, L)
    """

    def __init__(
        self,
        input_channels=8,
        hidden_dim=640,
        depth=6,
        num_targets=1,
        dropout=0.1
    ):
        super().__init__()

        self.num_targets = num_targets

        # local sequence encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_dim,
                kernel_size=13,
                padding=6
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=7,
                padding=3
            ),
            nn.GELU()
        )

        # sequence compression
        self.downsample = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=8,
                stride=2,
                padding=3
            ),
            nn.GELU()
        )

        # bidirectional Mamba blocks
        blocks = []

        for i in range(depth):

            blocks.append(
                Residual(
                    BiMambaBlock(
                        d_model=hidden_dim,
                        d_state=64,
                        d_conv=7,
                        expand=2,
                        headdim=80
                    )
                )
            )

            # intermediate regularization
            if i + 1 == depth // 2:

                blocks.append(
                    nn.Dropout(dropout)
                )

        self.sequence_model = nn.Sequential(
            *blocks
        )

        # final feature dropout
        self.dropout = nn.Dropout(
            dropout
        )

        # restore nucleotide resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dim,
                hidden_dim,
                kernel_size=8,
                stride=2,
                padding=3
            ),
            nn.GELU()
        )

        # prediction head
        self.head = nn.Conv1d(
            hidden_dim,
            num_targets,
            kernel_size=1
        )


    def forward(
        self,
        x
    ):

        seq_mask = (
            x.sum(dim=1) > 0
        ).float()


        x = self.encoder(x)

        x = self.downsample(x)


        # B,C,L -> B,L,C
        x = x.transpose(
            1,
            2
        )


        x = self.sequence_model(
            x
        )


        # B,L,C -> B,C,L
        x = x.transpose(
            1,
            2
        )


        x = self.dropout(
            x
        )


        x = self.upsample(
            x
        )


        logits = self.head(
            x
        )


        return logits, seq_mask