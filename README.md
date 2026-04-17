# AdaptFM

An interactive framework for annotating, training, running inference, and benchmarking 3D segmentation models. 

In AdaptFM you can:
- Test and benchmark 3D segmentation models on your data. We currently support CellposeSAM, MicroSAM, nnUNetv2, SAM-Med-3D, and SSVT.
- Create 3D annotations using a number of segmentation algorithms, including SAM2 for click-based segmentation and SAM3 for text-based segmentation
- Use those annotations to fine-tune/train models on your data
- Add new models or annotation algorithms

[Installation](#Installation) | [Get Started](Documentation/Standard) | [Customizing (Advanced Users)](Documentation/Customizations/customizations-main.md)

![Segmentation Demo](asset/demo_cropped.gif) ![SAM2 Demo](asset/SAM2_GIF.gif)

# Installation

Create a conda environment for AdaptFM using
```
conda create --name AdaptFM python=3.13
conda activate AdaptFM
```

Install PyTorch for your CUDA version using the selector [here](https://pytorch.org/get-started/locally/). If you don't know your CUDA version, run nvidia-smi in the terminal it will be at the top of the printout. 

Clone the repo and install required dependencies

```
pip install --upgrade pip
git clone https://github.com/Zhao-Lab-UW-DHO/AdaptFM.git
cd AdaptFM
pip install -e.
python -c "import AdaptFM; print('AdaptFM installed successfully')"
```

[SAM2](https://github.com/facebookresearch/sam2) and [SAM3](https://github.com/facebookresearch/sam3) have specific system requirements. Consequently, the are left as optional dependencies

```
pip install -e ".[sam2]"
pip install -e ".[sam3]"
```

## Using External Models with AdaptFM

**To avoid dependency conflicts, we recommend installing each model's package in its own conda environment.** Visit the below links to properly install the standard models

[nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/master)

[MicroSAM](https://github.com/computational-cell-analytics/micro-sam)

[CellposeSAM](https://github.com/mouseland/cellpose)

[SAM-Med-3D](https://github.com/uni-medical/sam-med3d)

Once you have created conda environments for each, navigate to AdaptFM > model > registry.py and add the path to the conda environment

```python
   "microSAM": MicroSAMSpec(
        name="microSAM",
        conda_env="/path/to/micro_sam/conda_environment", #replace with the path to your newly installed conda environment for microsam
        module_path="micro_sam.training",
        training_wrapper_path = "micro_sam.train_wrapper",
        inference_wrapper_path = "micro_sam.inference_wrapper"
    ),
```

## Using SSVT

SSVT is a model pretrained on roughly 180,000 organoid images. It is publicly avaialble for download on [hugging face](https://huggingface.co/hbakhtiar/SSVT_Organoids/tree/main). 
