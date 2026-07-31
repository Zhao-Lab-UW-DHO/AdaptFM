import argparse
import numpy as np
import os
from cellSAM import segment_cellular_image
from cellSAM import  get_model
import tifffile as tiff
import shutil
import numpy as np
import torch
import gc
import socket
import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D


def safe_segment(slice2d, model):
    try:
        mask,_,_ = segment_cellular_image(slice2d, model=model, device="cuda")

        if mask is None:
            return np.zeros(slice2d.shape, dtype=np.uint16)

        # --- ADD THIS ---
        if mask.shape != slice2d.shape:
            print(f"[WARN] Shape mismatch: input={slice2d.shape}, mask={mask.shape}")

        return mask

    except Exception as e:
        print(f"[WARN] CellSAM failed on slice with error: {e}")
        return np.zeros(slice2d.shape, dtype=np.uint16)


def segment_all_planes(image, model):

    Z, Y, X = image.shape
    masks = {'xy': [], 'xz': [], 'yz': []}

    # XY planes — iterate over Z
    for z in range(Z):
        masks['xy'].append(safe_segment(image[z, :, :], model))

    # XZ planes — iterate over Y
    for y in range(Y):
        masks['xz'].append(safe_segment(image[:, y, :], model))

    # YZ planes — iterate over X
    for x in range(X):
        masks['yz'].append(safe_segment(image[:, :, x], model))

    return {k: np.stack(v) for k, v in masks.items()}


def run_inference(test_dir, output_path):#     import multiprocessing as mp


    model = get_model(model='cellsam_extra')
    model = model.to("cuda")  
    for image_name in os.listdir(test_dir):
   # moves entire model to GPU
        
        image_path = os.path.join(test_dir,image_name)
        image = tiff.imread(image_path)

        try:

            results = segment_all_planes(image,model)

            indirect_aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
            indirect_aggregation_params['indirect_method']['dtform_method'] = 'edt' #Need to avoid dask within a subprocess

            segmentation3D, (probability3D, gradients3D) = uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(
                segmentations=[results['xy'], results['xz'], results['yz']],
                img_xy_shape=results['xy'].shape,
                precomputed_binary=None,
                params=indirect_aggregation_params,
                savefolder=None,
                basename=None
            )
            image_output_path = os.path.join(output_path,image_name)
            tiff.imwrite(image_output_path,segmentation3D)
        except Exception as e:
            print(f"Segmentation Failed for {image_name}")
        


    
    del image, results, model
    torch.cuda.empty_cache()

    gc.collect()
    gc.collect()


if __name__ =='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')


    args = parser.parse_args()

    run_inference(args.test_dir, args.output_path)


