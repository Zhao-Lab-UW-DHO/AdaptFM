<div align="center">

# AdaptFM

**A GUI Framework for 3D Foundation Model Segmentation, Interactive Image Annotation, and Benchmarking**

[Quick Start Guide](docs/quick-start.md) | [User Guide](docs/user-guide/annotations.md) | [Demo on Public Data](docs/testing-adaptfm.md) | [Customize AdaptFM](docs/customizing-adaptfm/index.md) | [Documentation](https://adaptfm.readthedocs.io/)

</div>

---

Built on napari, AdaptFM is a toolkit for running public 3d segmentation models on your data without computational experience. A number of utilities facilitate this:
- Create 3D annotations interactively using a wide selection of segmentation algorithms, including SAM2 click-based segmentation and SAM3 text-based segmentation
- Use those annotations to fine-tune/train models on your data
- Add new models or annotation algorithms
- Process input and output data 

| SAM3 Text-Based Annotation | SAM2 Click-Based Annotation |
| :---: | :---: |
| ![Segmentation Demo](asset/demo_cropped.gif) | ![SAM2 Demo](asset/SAM2_GIF.gif) |

> If you encounter installation errors, broken package dependencies, or other problems please open an [Issue](https://github.com/Zhao-Lab-UW-DHO/AdaptFM/issues).

## <a id="external-model-install"></a> Using Supported External Models with AdaptFM

AdaptFM supports a variety of external foundation models for medical imaging. These are optional; you only need to install the models you plan to use. The supported models are listed:

| Name | Paper | Code | VRAM Recommendations | Original Modality | Pretrained Model Datatype |
| --- | --- | --- | --- | --- | --- |
| nnUNet | [Link](https://www.nature.com/articles/s41592-020-01008-z) | [Link](https://github.com/MIC-DKFZ/nnUNet) | [Link](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md#how-to-use-the-new-presets) | General Biomedical Imaging | N/A |
| Merlin nnUnet | [Link](https://arxiv.org/abs/2406.06512) | [Link](https://github.com/ashwinkumargb/Merlin-nnUNet) | [Link](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md#how-to-use-the-new-presets) | CT | CT images with EHR diagnoses and radiology reports |
| CellSAM | [Link](https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3) | [Link](https://github.com/vanvalenlab/cellsam) | None listed | General Cell Imaging | tissue, cell culture, yeast, H&E and bacteria |
| SAMMed3D | [Link](https://arxiv.org/abs/2310.15161) | [Link](https://github.com/uni-medical/sam-med3d) | None listed | CT and MRI | organs, tissues, tumors, and blood vessels |
| CellposeSAM | [Link](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) | [Link](https://github.com/MouseLand/cellpose) | None listed | General Cell Imaging | Microbes, cultured cells, human tissues, and cell nuclei |
| microSAM | [Link](https://www.nature.com/articles/s41592-024-02580-4) | [Link](https://github.com/computational-cell-analytics/micro-sam) | [Link](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#usage-questions) | Light, electron, and X-ray microscopy (in various models) | Microbes, cells, organoids, organelles, nuclei, and tissues |
| BME-X | [Link](https://www.nature.com/articles/s41551-024-01283-7) | [Link](https://github.com/DBC-Lab/Brain_MRI_Enhancement) | None listed | MR Images | Brain |
| CT-FM | [Link](https://arxiv.org/abs/2501.09001) | [Link](https://github.com/project-lighter/CT-FM) | None listed | CT | Human anatomical structures |

AdaptFM also provides our own foundation model implementing a self-supervised vision transformer: [SSVT Documentation](docs/using-fms/ssvt.md).
 
## <a id="testing-adaptfm"></a>Testing AdaptFM

We have provided test data for AdaptFM on [Hugging Face](https://huggingface.co/datasets/hbakhtiar/AdaptFM_Testing/tree/main). Descriptions for each dataset are in the repository's README file.

### How to Download Data
Use 'include' to specify the dataset you want to download. The folder name should have /* at the end to download all contents in the folder. The folder name should be in quotes as below.  

> [!WARNING]
> The entire test dataset repository is over 100 GB. Make sure you have enough disk space before downloading everything at once, or download only specific dataset folders as needed

* **Download a specific dataset folder (e.g., BBBC024):**
    ```bash
    hf download hbakhtiar/AdaptFM_Testing --repo-type dataset --include "BBBC024/*" --local-dir "/path/to/your/folder"
    ```

* **Download the entire test dataset (~100 GB):**
    ```bash
    hf download hbakhtiar/AdaptFM_Testing --repo-type dataset --local-dir "/path/to/your/folder"
    ```

* **Download the SSVT Organoids model checkpoint:**
    ```bash
    hf download hbakhtiar/SSVT_Organoids --local-dir "/path/to/your/folder"
    ```

## <a id="contributing"></a>Contributing

AdaptFM is built to easily integrate new benchmarking metrics, segmentation foundation models, 2D annotation algorithms and more! To contribute these components, check out the [Customizing AdaptFM](docs/customizing-adaptfm/index.md) page which details the kind of contributions that fit seamlessly into AdaptFM. For feature enhancements, bug fixes, and general guidelines, refer to the [Contribution Guide](docs/user-guide/contributing.md).
