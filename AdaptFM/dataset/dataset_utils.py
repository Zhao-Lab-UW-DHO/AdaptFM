import json
import re
from pathlib import Path


# Function to construct required folder structure for nnunetv2
def construct_nnUNet_folders(base_dir, setID=1, setName="Organoids"):
    base = Path(base_dir)
    dataset_root = base / "nnUNet_raw" / f"Dataset{setID:03}_{setName}"

    (base / "nnUNet_preprocessed").mkdir(parents=True, exist_ok=True)
    (base / "nnUNet_results").mkdir(parents=True, exist_ok=True)

    paths = {
        "dataset_root": str(dataset_root),
        "imagesTr": str(dataset_root / "imagesTr"),
        "imagesTs": str(dataset_root / "imagesTs"),
        "labelsTr": str(dataset_root / "labelsTr"),
    }

    for name, path_str in paths.items():
        path_obj = Path(path_str)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path_str}")
        else:
            print(f"Directory already exists: {path_str}")

    return paths


# the .json file for nnunet needs to know the max number of channels that an image can have.
# for now we assume that all images (training and prediction) all have the same number of channels


def get_channel_dict(directory, setName, channel=0):
    # Pattern to match the filename convention
    pattern = rf"{setName}_\d{{3}}_000(\d+)\.tiff"

    if channel is not None:
        channel_names = {f"Channel {channel}": f"{channel}"}
        return channel_names

    num_channels = None
    for file_obj in Path(directory).iterdir():
        if file_obj.is_file():
            filename = file_obj.name
            match = re.match(pattern, filename)
            if match:
                channel_count = int(match.group(1))  # Extract the Y value
                # Update the largest Y value
                if num_channels is None or channel_count > num_channels:
                    num_channels = channel_count

    channel_names = {f"Channel {i}": f"{i}" for i in range(num_channels + 1)}
    return channel_names


#
# need to find the number of training cases for the json file
#


def count_unique_cases(directory, setName):
    # Pattern to match the filename convention and extract cases
    pattern = rf"{setName}_0*(\d+)_000\d+"

    unique_cases = set()

    for file_obj in Path(directory).iterdir():
        if file_obj.is_file():
            filename = file_obj.name
            match = re.match(pattern, filename)
            if match:
                case = int(match.group(1))  # Extract the case value
                unique_cases.add(case)

    return len(
        unique_cases
    )  # just need to know the number of unique training/testing cases


#
# function that loops through training and testing directories
#  creates JSON file for nnUNetv2
#   need the following pieces of info to create the file
#       -output directory for json file
#       - channel_names = can name them whatever, but need to know the max number of channels
#       - labels = background = 0, foreground = 1
#       - file_ending
#       - kwargs with dictionary specifying the training and test images
#
#
#


def write_nnUNet_json(base_dir, setName, setID, file_ending=".tiff", channel=0):

    formatted_setID = f"{int(setID):03d}"
    nnUNet_directory = str(
        Path(base_dir) / "nnUNet_raw" / f"Dataset{formatted_setID}_{setName}"
    )
    training_directory = str(Path(nnUNet_directory) / "imagesTr") + "/"
    # Assume for now that we can get this directly from the traiing directory

    channel_dict = get_channel_dict(
        training_directory, setName=setName, channel=channel
    )
    num_training = count_unique_cases(training_directory, setName=setName)

    # Creating binary, so can hardcode this

    labels_dict = {"background": 0, "Foreground": 1}

    # Hard code the file ending for now

    dataset_json = {
        "channel_names": channel_dict,
        "labels": labels_dict,
        "numTraining": num_training,
        "file_ending": file_ending,
    }

    with (Path(nnUNet_directory) / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, sort_keys=False, indent=4)
