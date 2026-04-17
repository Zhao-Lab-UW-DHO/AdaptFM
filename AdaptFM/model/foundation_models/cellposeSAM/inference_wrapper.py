import argparse
from cellpose import models 
import os
import tifffile as tiff
import SimpleITK as sitk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')

    args = parser.parse_args()
    test_dir = args.test_dir
    output_path = args.output_path
    checkpoint = args.checkpoint

    if checkpoint is not None:

        model = models.CellposeModel(
            gpu=True,
            pretrained_model=checkpoint
        )
    else:
        model = models.CellposeModel(gpu=True)
    
    # run model on test images
    for image_name in os.listdir(test_dir):

        image_path = os.path.join(test_dir,image_name)
        image = tiff.imread(image_path)

        masks, _, _ = model.eval(image,do_3D=True,channel_axis=3,z_axis=0)

        image_output_path = os.path.join(output_path,image_name)
        tiff.imwrite(image_output_path,masks)

if __name__ == "__main__":
    main()
