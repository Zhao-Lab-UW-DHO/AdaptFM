# Installing AdaptFM

AdaptFM uses multiple conda environments to manage its install. The environment needed to run AdaptFM built into napari requires the requirements listed in the pyproject.toml at the root of the repository as well as pytorch (torch and torchvision on PyPI).

AdaptFM's environment manager in napari on the backend makes calls to various install scripts found in `<repo_root>/AdaptFM/install` which can be inspected to see the requirements needed to run a particular model usually consisting of a package install, followed by a pytorch install, and sometimes a download of an available model checkpoint for inference written to a particular location. For interactive segmentation using SAM2 and SAM3, those install commands are made into the main conda environment alongside the AdaptFM software.

AdaptFM runs other environments for different models using subprocess such as `conda run -p <path_to_the_model_conda_environment> python <path_to_the_model_running_script.py>` or as a module call (with python -m).

## Manual Installation

Following the install scripts as described above with allow you to install the environments required to run models in AdaptFM. Other than installing the required packages including pytorch, AdaptFM will write the conda environment location in a .prefix file within a subdirectory of the home directory found with `echo "$HOME/.adaptfm/"` in one file per environment.

for example listing the AdaptFM directory after the environment manager makes an install of multiple models with:
```bash
cd $HOME/.adaptfm/; ls
```
will produce something like:
```bash
BME-X_adapt.prefix  cellpose_adapt.prefix  cellsam_adapt.prefix  CTFM_adapt.prefix  Merlin_nnUNet_adapt.prefix  micro-sam_adapt.prefix  nnUNet_adapt.prefix  pytorch_cmd.txt  usegment3d_adapt.prefix
```
pytorch_cmd.txt is the CLI command used to install pytorch into the environment for the AdaptFM install scripts
```bash
cat pytorch_cmd.txt 
```
gives `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132` for a system with CUDA version >13.1.

and the prefix files should give the same conda prefix found in `conda info --envs`
```bash
cat cellsam_adapt.prefix 
```
gives `/home/<username>/<conda install folder>/envs/<env name>`

critically **the names of the conda environments are important** and found in the install script, as the conda environment name is used in the AdaptFM model registry to recognize the environment to run.
## Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
WARNING: Could not load the Qt platform plugin "xcb" in "" even though it was found.
</summary>
<br/>

This happens when there are missing display libraries. For our system the fix was to install with the AdaptFM environment loaded: `conda install xcb-util-cursor`,
then set the linux environment variable before launching `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; python -m AdaptFM.dev_launch`
</details>

<details>
<summary>
When opening an image: OpenGL error attempted to retrieve context when no valid context
</summary>
<br/>

We found this issue when testing AdaptFM on Windows Subsystem for Linux and believe it to be a catch-all error pointing to the graphics pipeline from WSL to Windows being quite fragile. Enforcing qt6 with QT_API=pyqt6 on WSL fixes this issue on our system, but we have not found a solution that fixes this error for use of the apptainer/Singularity container in WSL
</details>

## Running AdaptFM in a container

In the container folder of the repository we provide a file `AdaptFM_apptainer.def` that defines an apptainer/singularity container using an NVIDIA 12.6 CUDA Ubuntu image installing similar dependencies to the requirements of [napari-xpra](https://github.com/napari/napari/pkgs/container/napari-xpra). With [apptainer installed](https://apptainer.org/docs/admin/main/installation.html), the `build_and_run.sh` script in the same folder demnstrates the creation of a ~6GB large container image and using it to launch AdaptFM. The ability to build the container depends on system restrictions like administrator granted permissions and building the container may require setting environment variables such as APPTAINER_TMPDIR, APPTAINER_CACHEDIR to build successfully.

The details of the napari-xpra container combined with our container's def file should allow an advanced user to customize our image to run AdaptFM with XPRA if required.