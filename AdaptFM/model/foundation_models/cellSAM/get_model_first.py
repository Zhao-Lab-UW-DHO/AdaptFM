"""First time cellsam runs it needs the access token to get the model
this is run once at the time of install so users have the model saved locally
"""

import argparse
import os

from cellSAM import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access_token")
    args = parser.parse_args()

    os.environ.update({"DEEPCELL_ACCESS_TOKEN": f"{args.access_token}"})
    model = get_model(model="cellsam_extra")

    print("Successfully installed model")


if __name__ == "__main__":
    main()
