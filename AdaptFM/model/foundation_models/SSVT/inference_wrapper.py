import torch
import tifffile
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from AdaptFM.SSVT.Vit_class import ViTEncoder3D,MAE3DSegmentation,MAE3DLinearProbeDecoder,MAE3DUNetDecoderBig
from AdaptFM.SSVT.utils import read_tiff,sliding_window_inference


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')
    output_path = Path(output_path)

    args = parser.parse_args()
    test_dir = args.test_dir
    output_path = args.output_path
    checkpoint = args.checkpoint

    #baseline SSVT is hard-coded based on how it was pretrained
    device = 'cuda'

    encoder = ViTEncoder3D(patch_size = (2,16,16),
                           embed_dim = 768,
                           depth = 8,
                           num_heads=8).to(device)
    
    #initialize decoder based on fixed parameters

    dummy_patch = torch.zeros(1, 1, *(4,128,128), device=device)
    with torch.no_grad():
        _, patch_grid = encoder(dummy_patch, return_grid=True)


    decoder = MAE3DUNetDecoderBig(embed_dim=768,patch_grid=patch_grid) #change to MAE3DUNetDecoder if you want the UNET decoder for more complex segmentation tasks
    model = MAE3DSegmentation(encoder,decoder).to(device)

    #load model checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    torch.set_grad_enabled(False)

    # ---- Iterate over test volumes ----
    test_paths = sorted(Path(test_dir).glob("*.tiff"))
    for path in tqdm(test_paths, desc="Testing"):
        vol = read_tiff(path)

        seg_np, _ = sliding_window_inference(
            model,
            vol,
            patch_size=(4,128,128),
            stride=(16,32,32),
            device=device,
            threshold=0.5
        )

        out_path = output_path / path.name
        tifffile.imwrite(out_path, seg_np)


if __name__ == "__main__":
    main()
