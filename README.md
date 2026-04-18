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
pip install -e .
python -c "import AdaptFM; print('AdaptFM installed successfully')"
```

[SAM2](https://github.com/facebookresearch/sam2) and [SAM3](https://github.com/facebookresearch/sam3) have specific system requirements. Consequently, you can optionally install them using the below commands. AdaptFM will still work if you do not install these tools. 

**If you install SAM3 below and do not have a supported CUDA version you will not be able to use AdaptFM**

```
adaptfm-install-sam2
adaptfm-install-sam3
```
**If using SAM3 you must request access to their checkpoints through hugging face. These cannot be auto-downloaded**. Once you have downloaded checkpoints, make a folder called 'checkpoints' in AdaptFM > segmentation > sam3 and add the checkpoint files there.  

Once installed you can launch AdaptFM from the terminal using

```
python -m AdaptFM.dev_launch
```


## Using External Models with AdaptFM

**To avoid dependency conflicts, we recommend installing each model's package in its own conda environment.**

Below are installations that we found worked. If you run into installation issues, visit the below links for more detailed instructions. 

**You only need to install model you want to use.** AdaptFM will work if you don't install the below models

[nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/master) - Note that nnUNet is currently experiencing [a bug](https://github.com/MIC-DKFZ/nnUNet/issues/3009) that will prevent users from training.  

```
conda create --name nnUNet_adapt python=3.10
conda activate nnUNet_adapt
```

Install the proper pytorch verison https://pytorch.org/get-started/locally/ in your new environment. Then install nnunetv2 with

```
pip install nnunetv2
```


[MicroSAM](https://github.com/computational-cell-analytics/micro-sam) 

```
conda create -c conda-forge -n micro-sam_adapt python=3.10 micro_sam
```

[CellposeSAM](https://github.com/mouseland/cellpose)

```
conda create --name cellpose_adapt python=3.10
conda activate cellpose_adapt
python -m pip install cellpose
```
Again, install the proper pytorch version in your cellpose_adapt environment with https://pytorch.org/get-started/locally/

[SAM-Med-3D](https://github.com/uni-medical/sam-med3d) - once you have created the conda environment and checkpoint, use the below command to install the repo

```
conda create --name sammed3d_adapt python=3.10 -y
conda activate sammed3d_adapt

pip install uv

uv pip install opencv-python-headless matplotlib \
    prefetch_generator monai edt surface-distance medim numpy

git clone https://github.com/uni-medical/SAM-Med3D.git
cd SAM-Med3D
```

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

SSVT is supported directly within AdaptFM. It does not require a separate conda environment or installation. You only need to update AdaptFM > model > registry.py with the location of the AdaptFM conda environment.

Once downloaded you can follow our [fine-tuning instructions](Documentation/Standard/Fine-tuning-models.md) to build a model for a specific downstream segmentation task. 
