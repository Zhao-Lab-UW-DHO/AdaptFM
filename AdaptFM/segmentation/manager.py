from dataclasses import dataclass
import os
from datetime import datetime
import tifffile as tiff
import json
from AdaptFM.segmentation.fourier.nuc_seg import run_nuclear_segmentation
from pathlib import Path

@dataclass
class AutoSegParams:
    percentile: float = 90.0
    max_freq: float = 20.0
    frequency_step: float = 1.0
    sigma: float = 0.3
    remove_background: bool = True


class SegmentationManager:
    def __init__(self, volume_manager):
        self.vm = volume_manager
        self.params = AutoSegParams()
        self.last_segmentation = None
        self.history = []

    # -------------------------
    # Public API
    # -------------------------

    def auto_segment(self, *, record=True):
        """
        Explicit segmentation trigger.
        """
        img = self.vm.get_eager()  # segmentation should be eager

        seg = run_nuclear_segmentation(
            volume=img,
            percentile=self.params.percentile,
            max_freq=self.params.max_freq,
            frequency_step=self.params.frequency_step,
            sigma=self.params.sigma,
            remove_background=self.params.remove_background,
        )

        self.last_segmentation = seg

        if record:
            self.history.append({
                "params": self.params.__dict__.copy(),
                "path": self.vm.path,
                "shape": seg.shape
            })

        return seg

    def save_for_training(self, save_dir, filename_base=None, manual_seg=None):

            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # --- filenames ---
            if filename_base is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"volume_{timestamp}"

            vol_path = str(Path(save_dir) / f"{filename_base}.tiff")
            seg_path = str(Path(save_dir) / f"{filename_base}_seg.tiff")

            # --- get data ---
            volume = self.vm.get_eager()
            segmentation = manual_seg if manual_seg is not None else self.last_segmentation

            # --- save volume & segmentation ---
            tiff.imwrite(vol_path, volume.astype(volume.dtype))
            tiff.imwrite(seg_path, segmentation.astype(segmentation.dtype))

            # --- save metadata ---


            return vol_path, seg_path
