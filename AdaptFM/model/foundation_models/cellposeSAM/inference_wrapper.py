import argparse
from pathlib import Path
from cellpose import models 
import os
import tifffile as tiff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')

    args = parser.parse_args()
    test_dir = args.test_dir
    output_path = args.output_path
    checkpoint = args.checkpoint
    
    if checkpoint == "None":
        checkpoint = None

    if checkpoint is not None:

        model = models.CellposeModel(
            gpu=True,
            pretrained_model=checkpoint
        )
    else:
        model = models.CellposeModel(gpu=True)
    
    # run model on test images
    for image_obj in Path(test_dir).iterdir():
        if not image_obj.is_file():
            continue
            
        image_name = image_obj.name
        image_path = str(image_obj)
        image = tiff.imread(image_path)

        masks, _, _ = model.eval(image, do_3D=True, channel_axis=3, z_axis=0)

        image_output_path = str(Path(output_path) / image_name)
        tiff.imwrite(image_output_path, masks)

if __name__ == "__main__":
    main()
