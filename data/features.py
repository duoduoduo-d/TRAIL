import torch


BASE_INDEX = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3
}


def calculate_offset(
    gene_type,
    tx_size,
    cds_start,
    max_utr5_len=1500
):
    """
    Map transcript coordinate to model coordinate.
    """

    if gene_type == "lncRNA":

        offset = (
            max_utr5_len
            -
            0.05 * tx_size
        )

    else:

        offset = (
            max_utr5_len
            -
            cds_start
        )

    return int(offset)



def encode_transcript(
    sequence,
    gene_type,
    tx_size,
    cds_start,
    cds_end,
    splice_sites=None,
    max_length=13200,
    max_utr5_len=1500
):
    """
    Encode transcript into 8-channel representation.

    Channels:
        0: A
        1: T
        2: C
        3: G
        4: splice site
        5: 5UTR
        6: CDS
        7: 3UTR

    Returns:
        x:
            (8, max_length)

        offset:
            transcript coordinate offset

        end_pos:
            transcript end position
    """


    sequence = (
        str(sequence)
        .upper()
        .replace("U", "T")
    )


    offset = calculate_offset(
        gene_type,
        tx_size,
        cds_start,
        max_utr5_len
    )


    x = torch.zeros(
        (
            8,
            max_length
        ),
        dtype=torch.float32
    )


    # =========================================
    # nucleotide encoding
    # =========================================

    for i, nt in enumerate(sequence):

        model_pos = offset + i

        if (
            0 <= model_pos <
            max_length
        ):

            if nt in BASE_INDEX:

                x[
                    BASE_INDEX[nt],
                    model_pos
                ] = 1.0



    # =========================================
    # splice site channel
    # =========================================

    if splice_sites is not None:


        if isinstance(
            splice_sites,
            str
        ):

            if splice_sites.strip() != "":

                sites = splice_sites.split(",")

            else:

                sites = []


        elif isinstance(
            splice_sites,
            list
        ):

            sites = splice_sites


        else:

            sites = []



        for site in sites:

            try:

                site = int(site)

            except:

                continue


            model_pos = (
                offset
                +
                site
            )


            if (
                0 <= model_pos <
                max_length
            ):

                x[
                    4,
                    model_pos
                ] = 1.0



    # =========================================
    # transcript region annotation
    # =========================================


    if gene_type == "protein_coding":

        cds_start_pos = offset + cds_start
        cds_end_pos = offset + cds_end
        tx_end_pos = offset + tx_size


        # 5UTR
        x[
            5,
            max(0, offset):
            min(max_length, cds_start_pos)
        ] = 1.0


        # CDS
        x[
            6,
            max(0, cds_start_pos):
            min(max_length, cds_end_pos)
        ] = 1.0


        # 3UTR
        x[
            7,
            max(0, cds_end_pos):
            min(max_length, tx_end_pos)
        ] = 1.0



    end_pos = (
        offset
        +
        tx_size
    )


    return (
        x,
        offset,
        end_pos
    )