# Installing AdaptFM

## Quick Start Install

If your system meets the requirements and prerequisites described in the [quickstart guide](quick-start.md), please use that guide to install and launch AdaptFM.

## Install Details

AdaptFM uses multiple Conda environments to manage its install. The environment needed to run the main AdaptFM GUI system built into napari requires the requirements listed in the pyproject.toml at the root of the repository as well as PyTorch (torch and torchvision on PyPI preferably with GPU install described [on the PyTorch local installation helper](https://pytorch.org/get-started/locally/)).

AdaptFM's model installer in napari on the backend makes calls to various install scripts found in `<repo_root>/AdaptFM/install` which can be inspected to see the requirements needed to run a particular model. The requirements usually consist of a package install, followed by a PyTorch install, and sometimes a download of an available model checkpoint for inference written to a particular location. For interactive segmentation using SAM2 and SAM3, those install commands are made into the main Conda environment alongside the AdaptFM software.

AdaptFM runs other environments for different models using subprocess such as `conda run -p <path_to_the_model_conda_environment> python <path_to_the_model_running_script.py>` or as a module call (with python -m).

## Windows Installation

You can use AdaptFM on Windows in 2 ways:

1. Through Windows Subsystem For Linux (Recommended)
2. [Native Windows](#manual-installation) - to install, follow the same steps as our manual installation instructions

***The best way to use AdaptFM on a Windows systems is with [Windows Subsystem For Linux 2](https://learn.microsoft.com/en-us/windows/wsl/install). Certain AdaptFM features (e.g. SAM2 and SAM3) require a linux system. Installation on Native Windows is possible, but not recommended, as these core features will not be available.***

Use the above link with step-by-step instructions on installing WSL2. Briefly, it involves running the below command as an administrator from the PowerShell command line

```
wsl --install
```

Once you have succesfully installed WSL2, you can use our installer script to install AdaptFM

**Clone the repository and install**
```bash
git clone https://github.com/Zhao-Lab-UW-DHO/AdaptFM.git
cd AdaptFM
chmod +x install.sh launch.sh
bash install.sh
```
Then, from the main AdaptFM folder, start the program by running `bash launch.sh` 

## Manual Installation

You can follow the below steps to install AdaptFM on native Windows or to install AdaptFM manually on a different system. While this option is available we recommend using WSL2 for AdaptFM, as some AdaptFM features require a Linux system and will be unavailable in native Windows.

With Conda and Git, the following set of commands will install AdaptFM without PyTorch:

```
conda create --name AdaptFM python=3.12 -y
conda activate AdaptFM
git clone https://github.com/Zhao-Lab-UW-DHO/AdaptFM.git
cd AdaptFM
pip install -e .
```

Then, since PyTorch must be installed based on the Compute Platform of your own system: use [the PyTorch local installation helper](https://pytorch.org/get-started/locally/) to find the `pip3 install` relevant to your system. Choosing the correct OS, Package, Python, and the Compute Platform of your GPU (typically found by looking at the CUDA Version of `nvidia-smi`) selected. Copy and paste the PyTorch `pip3 install` command when prompted from running:
```bash
adaptfm-set-pytorch
``` 
This will install PyTorch into AdaptFM as well as save the hardware specific install for use in automatic installation of External Models.

Then to launch AdaptFM, navigate to the main AdaptFM folder, use either of the below scripts:

```
bash launch.sh
```
or
```
python -m AdaptFM.dev_launch
```

## Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
WARNING: Could not load the Qt platform plugin "xcb" in "" even though it was found.
</summary>
<br/>

This happens when there are missing display libraries. For our system, the fix was to install with the AdaptFM environment loaded: `conda install xcb-util-cursor`,
then set the Linux environment variable before launching e.g. `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; python -m AdaptFM.dev_launch` or edit `launch.sh` to add `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` preceding python -m AdaptFM.dev_launch. 
</details>

<details>
<summary>
When opening an image: OpenGL error attempted to retrieve context when no valid context.
</summary>
<br/>

We found this issue when testing AdaptFM on Windows Subsystem for Linux and believe it to be a catch-all error pointing to the graphics pipeline from WSL to Windows being quite fragile. Enforcing qt6 with QT_API=pyqt6 on WSL fixes this issue on our system, but we have not found a solution that fixes this error for use of the apptainer/Singularity container in WSL.
</details>

<details>
<summary>
ModuleNotFoundError: No module named 'AdaptFM'
</summary>
<br/>
 
This happens when the AdaptFM code repository is not in the PYTHONPATH environment variable which is used by python to search when importing modules. To fix this, ensure that AdaptFM is launched from the root of the AdaptFM repository, e.g. running `ls` or `dir` should show the files `pyproject.toml` and `launch.sh` as well as folders `AdaptFM` and `docs`.
</details>

<details>
<summary>
Model Installer failing to install models
</summary>
<br/>
 
This can happen if running an installation on a non-linux-based system. You can manually install models by navigating to the AdaptFM folder > pyproject.toml and locate the install script for the model. It should have the format 

```
conda activate AdaptFM
adaptfm-install-MODEL
```

Following the install scripts as described above will allow you to install the environments required to run models in AdaptFM. If that still doesn't work, you can navigate to the install script (AdaptFM > install) to see how we installed the model. AdaptFM generally writes the Conda environment location in a `.prefix` file within a subdirectory of the home directory found with `echo "$HOME/.adaptfm/"` in one file per environment.

For example, listing the AdaptFM directory after the model installer makes an install of multiple models with:
```bash
cd $HOME/.adaptfm/; ls
```
will produce something like:
```bash
BME-X_adapt.prefix  cellpose_adapt.prefix  cellsam_adapt.prefix  CTFM_adapt.prefix  Merlin_nnUNet_adapt.prefix  micro-sam_adapt.prefix  nnUNet_adapt.prefix  pytorch_cmd.txt  usegment3d_adapt.prefix
```
pytorch_cmd.txt is the CLI command used to install PyTorch into the environment for the AdaptFM install scripts
```bash
cat pytorch_cmd.txt 
```
gives `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132` for a system with CUDA version >13.1.

and the prefix files should give the same Conda prefix found in `conda info --envs`
```bash
cat cellsam_adapt.prefix 
```
gives `/home/<username>/<conda install folder>/envs/<env name>`

Critically **the names of the Conda environments are important** and found in the install script, as the Conda environment name is used in the AdaptFM model registry to recognize the environment to run.


</details>


## Running AdaptFM in a Container

In the container folder of the repository we provide a file `AdaptFM_apptainer.def` that defines an apptainer/singularity container using an NVIDIA 12.6 CUDA Ubuntu image installing similar dependencies to the requirements of [napari-xpra](https://github.com/napari/napari/pkgs/container/napari-xpra). With [apptainer installed](https://apptainer.org/docs/admin/main/installation.html), the `build_and_run.sh` script in the same folder demonstrates the creation of a ~6 gigabytes large container image and using it to launch AdaptFM. The ability to build the container depends on system restrictions like administrator granted permissions and building the container may require setting environment variables such as APPTAINER_TMPDIR, APPTAINER_CACHEDIR to build successfully.

The details of the napari-xpra container combined with our container's def file should allow an advanced user to customize our image to run AdaptFM with XPRA if required.