# Imports
import torch
from lighter_zoo import SegResNet
from monai.transforms import (
    Compose, LoadImage, EnsureType, Orientation,
    ScaleIntensityRange, CropForeground, Invert,
    Activations, AsDiscrete, KeepLargestConnectedComponent,
    SaveImage
)
from monai.inferers import SlidingWindowInferer
import argparse
import os

def main(
        test_dir,
        output_path
):


    model = SegResNet.from_pretrained(
        "project-lighter/whole_body_segmentation",
        force_download=True
    )
    
    print('Model Loaded')


        # Configure sliding window inference
    inferer = SlidingWindowInferer(
        roi_size=[96, 160, 160],  # Size of patches to process
        sw_batch_size=2,          # Number of windows to process in parallel
        overlap=0.625,            # Overlap between windows (reduces boundary artifacts)
        mode="gaussian"           # Gaussian weighting for overlap regions
    )


        # Preprocessing pipeline
    preprocess = Compose([
        LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
        EnsureType(),                         # Ensure correct data type
        Orientation(axcodes="RAS"),           # Standardize orientation
        # Scale intensity to [0,1] range, clipping outliers
        ScaleIntensityRange(
            a_min=-1024,    # Min HU value
            a_max=2048,     # Max HU value
            b_min=0,        # Target min
            b_max=1,        # Target max
            clip=True       # Clip values outside range
        ),
    ])

    # Postprocessing pipeline
    postprocess = Compose([
        Activations(softmax=True),              # Apply softmax to get probabilities
        AsDiscrete(argmax=True, dtype=torch.int32),  # Convert to class labels
        KeepLargestConnectedComponent(),        # Remove small disconnected regions
        # Save the result
        SaveImage(output_dir=output_path)
    ])


    for image_name in os.listdir(test_dir):

        print(image_name)
            
        image_path = os.path.join(test_dir,image_name)
        input_tensor = preprocess(image_path)

        with torch.no_grad():
            output = inferer(input_tensor.unsqueeze(dim=0),model)[0]

        output.applied_operations = input_tensor.applied_operations
        output.affine = input_tensor.affine


        result = postprocess(output)
            

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')


    args = parser.parse_args()

    main(args.test_dir,args.output_path)