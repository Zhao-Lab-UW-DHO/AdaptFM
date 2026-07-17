import glob
import os
import random
import tifffile as tiff
import shutil
from pathlib import Path
from AdaptFM.dataset.dataset_utils import construct_nnUNet_folders, write_nnUNet_json

class DatasetManager:
    def __init__(self, folder=None):
        self.samples = []  # list of dicts as above
        self.folder = []
        if folder:
            self.folder = folder

            self.load_from_folder(folder)

    def load_from_folder(self, folder):
        self.folder = folder
        image_paths = sorted(Path(folder).glob("*.tif*"))
        for img_path in image_paths:
            if img_path.name.endswith("_seg.tiff"):
                continue

            mask_path = img_path.with_name(f"{img_path.stem}_seg.tiff")

            if not mask_path.exists():
                continue

            self.samples.append({
                "id": img_path.stem,
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
        
        os.makedirs(out_folder,exist_ok=True)
        print(params)
        if framework =='nnunet':
            return self._export_nnunet(out_folder,params=params)
        else:
            return self._export_simple_pairs(out_folder,framework)
        
    
    def _export_nnunet(self,out_folder,file_ending='.tiff',channel=0,params=None):

        print(params)
        setID = params['Set ID']
        setName = params['Set Name']

        paths = construct_nnUNet_folders(
            out_folder, setID=setID, setName=setName
        )

        imagesTr = paths['imagesTr']
        labelsTr = paths['labelsTr']

        for idx, s in enumerate(self.samples):

            case_id = f"Organoids_{idx:03d}"

            img_dst = os.path.join(imagesTr, f"{case_id}_0000.tiff")
            lbl_dst = os.path.join(labelsTr, f"{case_id}.tiff")

            shutil.copy(s["image"], img_dst)
            shutil.copy(s["mask"], lbl_dst)

        write_nnUNet_json(
            base_dir = out_folder,
            setName=setName,
            setID = setID,
            file_ending=file_ending,
            channel=channel
            
        )

        return out_folder / f"nnUNet_raw/Dataset{setID:03}_{setName}"


    def _export_simple_pairs(self, out_folder, framework):
        img_dir = os.path.join(out_folder, "images")
        msk_dir = os.path.join(out_folder, "masks")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        for s in self.samples:
            shutil.copy(s["image"], os.path.join(img_dir, os.path.basename(s["image"])))
            shutil.copy(s["mask"], os.path.join(msk_dir, os.path.basename(s["mask"])))

        return Path(out_folder)

        
