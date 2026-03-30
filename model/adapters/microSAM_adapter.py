import shutil
from pathlib import Path
import os

class MicroSAMDatasetAdapter:
    def prepare(self, dataset_manager, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

        training_dir = output_dir / "training"
        seg_dir = output_dir / "segmentations"

        training_dir.mkdir(exist_ok=True)
        seg_dir.mkdir(exist_ok=True)

        for s in dataset_manager.samples:
            # --- image destination ---
            img_name = Path(s["image"]).name
            shutil.copy(s["image"], training_dir / img_name)

            # --- mask destination (remove '_seg') ---
            mask_name = Path(s["mask"]).name
            shutil.copy(s["mask"], seg_dir / mask_name)

        # IMPORTANT: MICROSAM NEEDS TRAINING AND SEGMENTATIONS IN THE SAME ORDER

        raw_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir)]
        label_paths = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]

        # Map raw filename → full path
        raw_dict = {
            os.path.basename(p): p
            for p in raw_paths
        }

        # Map *base name* (without _seg) → label path
        label_dict = {}
        for p in label_paths:
            fname = os.path.basename(p)
            if fname.endswith("_seg.tiff"):
                base_name = fname.replace("_seg.tiff", ".tiff")
                label_dict[base_name] = p

        # Intersection = valid paired samples
        common = sorted(set(raw_dict.keys()) & set(label_dict.keys()))

        valid_raw_paths = []
        valid_label_paths = []

        for fname in common:
            valid_raw_paths.append(raw_dict[fname])
            valid_label_paths.append(label_dict[fname])

        

            return output_dir
