import torch
import argparse


def main(path):

    ckpt = torch.load(
        path,
        map_location="cpu"
    )

    print("=" * 50)
    print("Checkpoint information")
    print("=" * 50)

    print("\nTargets:")
    for i, t in enumerate(ckpt["targets"]):
        print(f"{i}: {t}")


    print("\nConfig:")
    print(ckpt["config"])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    args = parser.parse_args()


    main(
        args.checkpoint
    )