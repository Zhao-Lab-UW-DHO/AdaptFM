import glob
import os
import random
import tifffile as tiff
import shutil
import json
from pathlib import Path
import SimpleITK as sitk
from AdaptFM.dataset.dataset_utils import construct_nnUNet_folders, write_nnUNet_json

class DatasetManager:
    def __init__(self, folder=None):
        self.samples = []  # list of dicts as above
        self.folder = []
        if folder:
            self.folder = folder

            self.load_from_folder(folder)

    def load_from_folder(self, folder):
        self.folder = Path(folder)

        # Supported image extensions
        exts = ["*.tif", "*.tiff", "*.nii", "*.nii.gz"]

        # Collect all matching files
        image_paths = []
        for ext in exts:
            image_paths.extend(self.folder.glob(ext))

        image_paths = sorted(image_paths)

        for img_path in image_paths:
            filename = img_path.name

            # Handle multi-dot extensions
            if filename.endswith(".nii.gz"):
                ext = ".nii.gz"
                stem = filename[:-7]
            else:
                ext = img_path.suffix
                stem = img_path.stem

            if stem.endswith("_seg"):
                continue
                
            mask_path = img_path.with_name(f"{stem}_seg{ext}")

            if not mask_path.exists():
                continue

            self.samples.append({
                "id": stem,
                "image": str(img_path),
                "mask": str(mask_path)
            })

    def iter_samples(self, shuffle=True):
        samples = self.samples.copy()
        if shuffle:
            random.shuffle(samples)
        for s in samples:
            img = tiff.imread(s["image"])
            mask = tiff.imread(s["mask"])

            yield img, mask

    def export_for_framework(self, framework="nnunet", out_folder=None,params=None):
        # copy files to nnunet folder structure or return lists of paths
        
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        
        print(params)
        if framework =='nnunet':
            return self._export_nnunet(out_folder,params=params)
        else:
            return self._export_simple_pairs(out_folder,framework)
        
    
    def _export_nnunet(self,out_folder,file_ending='.tiff',channel=0,params=None):

        
        setID = int(params['Set ID'])
        setName = params['Set Name']

        paths = construct_nnUNet_folders(
            out_folder, setID=setID, setName=setName
        )

        imagesTr = paths['imagesTr']
        labelsTr = paths['labelsTr']

        name_mapping= {}

        for idx, s in enumerate(self.samples):

            case_id = f"{setName}_{idx:03d}"

            img_dst = str(Path(imagesTr) / f"{case_id}_0000.tiff")
            lbl_dst = str(Path(labelsTr) / f"{case_id}.tiff")

            shutil.copy(s["image"], img_dst)
            shutil.copy(s["mask"], lbl_dst)

            name_mapping[s["image"]] = img_dst

        json_path = os.path.join(imagesTr, "name_mapping.json")
        with open(json_path, "w") as f:
            json.dump(name_mapping, f, indent=4)

        write_nnUNet_json(
            base_dir = out_folder,
            setName=setName,
            setID = setID,
            file_ending=file_ending,
            channel=channel
            
        )

        return out_folder 


    def _export_simple_pairs(self, out_folder, framework):
        out_folder_path = Path(out_folder)
        img_dir = out_folder_path / "images"
        msk_dir = out_folder_path / "masks"

        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for s in self.samples:
            shutil.copy(s["image"], img_dir / Path(s["image"]).name)
            shutil.copy(s["mask"], msk_dir / Path(s["mask"]).name)

        return out_folder_path

        
