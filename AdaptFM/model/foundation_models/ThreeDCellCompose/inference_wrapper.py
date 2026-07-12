from ThreeDCellComposer.ThreeDCellComposer import ThreeDCellComposer
import os
from pathlib import Path
import argparse
import traceback
import json

import ast

def normalize_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            # Try Python literal parsing
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return val
        except Exception:
            pass

        # Fallback: comma-delimited
        return [v.strip() for v in x.split(",")]

    raise ValueError(f"Invalid list format: {x}")


def run_cellcompose(input_dir,
                    output_dir,
                    params):
    
    input_dir = Path(input_dir)
    output_dir= Path(output_dir)

    os.environ["DEEPCELL_ACCESS_TOKEN"]=params['DeepCell_token']

    nucleus_channel_marker_list = normalize_list(params['nuclear_channel_list'])
    cytoplasm_channel_marker_list = normalize_list(params['cytoplasm_channel_list'])
    membrane_channel_marker_list = normalize_list(params['membrane_channel_list'])

    print(nucleus_channel_marker_list)

    for image_path in input_dir.iterdir():
        try:
            print(f"Processing: {image_path}") 
            result = ThreeDCellComposer(
                image_path,
                nucleus_channel_marker_list,
                cytoplasm_channel_marker_list,
                membrane_channel_marker_list,
                "deepcell",
                (1,1,1)
            )

        except Exception as e:
            print(f"Error processing {image_path}")
            print(traceback.format_exc())  # full traceback
            continue

    print("Batch processing complete.")
    return


    




if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')
    parser.add_argument('--params')

    args = parser.parse_args()
    params = json.loads(args.params)


    run_cellcompose(input_dir=args.test_dir,
                    output_dir=args.output_path,
                    params=params)