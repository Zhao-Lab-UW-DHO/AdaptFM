import shutil
from pathlib import Path


class MicroSAMDatasetAdapter:
    def prepare(self, dataset_manager, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

        training_dir = output_dir / "training"
        seg_dir = output_dir / "segmentations"

        training_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)

        for s in dataset_manager.samples:
            # --- image destination ---
            img_name = Path(s["image"]).name
            shutil.copy(s["image"], training_dir / img_name)

            # --- mask destination (remove '_seg') ---
            mask_name = Path(s["mask"]).name
            shutil.copy(s["mask"], seg_dir / mask_name)

        # IMPORTANT: MICROSAM NEEDS TRAINING AND SEGMENTATIONS IN THE SAME ORDER

        raw_paths = [str(f) for f in training_dir.iterdir() if f.is_file()]
        label_paths = [str(f) for f in seg_dir.iterdir() if f.is_file()]

        # Map raw filename → full path
        raw_dict = {Path(p).name: p for p in raw_paths}

        # Map *base name* (without _seg) → label path
        label_dict = {}
        for p in label_paths:
            path_obj = Path(p)
            fname = path_obj.name
            if path_obj.stem.endswith("_seg") and path_obj.suffix.lower() in (
                ".tif",
                ".tiff",
            ):
                base_name = path_obj.stem.replace("_seg", "") + ".tiff"
                label_dict[base_name] = p

        # Intersection = valid paired samples
        common = sorted(set(raw_dict.keys()) & set(label_dict.keys()))

        valid_raw_paths = []
        valid_label_paths = []

        for fname in common:
            valid_raw_paths.append(raw_dict[fname])
            valid_label_paths.append(label_dict[fname])

            return output_dir
