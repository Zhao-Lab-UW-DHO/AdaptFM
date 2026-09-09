# Quick Start Guide

Welcome to AdaptFM! This guide covers the fastest way to install the base framework and launch the GUI. 

**Note:** If you are installing AdaptFM on a custom hardware setup (e.g. with an AMD GPU) or want to use Apptainer/Singularity containers, please skip this page and follow our [Extended Installation](install-details.md).

---

## Prerequisites

AdaptFM requires [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) to be installed, and the ability for your system to install [PyTorch](https://pytorch.org/get-started/locally/) with a non-CPU Compute Platform.

To use our automated installation script (`install.sh`) your system must be running Linux and have an NVIDIA GPU installed and working (with nvidia-smi available in the command line interface). If you try the install and your machine does not meet these requirements try the documentation for [Extended Installation Details](install-details.md). We have found non NVIDIA GPU systems to be less optimized for AdaptFM and it's integrated components.

AdaptFM should be installable on any system capable of running a modern version of PyTorch (with GPU integration for practical reasons as well as our software's calls to the torch.cuda library assuming a GPU). AdaptFM is extensively tested on Ubuntu LTS with enterprise NVIDIA GPUs, and has been tested to run on the default image for Windows Subsystem for Linux 2 (WSL2) with both enterprise and consumer NVIDIA GPUs, including NVIDIA RTX 3090/4090. With a manual installation, we have found AdaptFM to have functionality on Arch Linux: 6.18-lts kernel using both Wayland and an AMD Radeon RX 9070 XT GPU, and on Windows 11 natively without GPU integration (not recommended).

The limitations of using a consumer GPU for AdaptFM are only the amount of GPU VRAM available to run large segmentation models.

NVIDIA GPU Driver versions with CUDA >= 12.6 are recommended as older versions are not supported in external code repositories integrated into AdaptFM. 

---

## Installation

The easiest way to install AdaptFM is with our automated installer in the command line interface of a Linux computer system. Windows 11 Users can install and run [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) within Windows for a Linux system that works with the installer.

**Clone the repository and install**
```bash
git clone https://github.com/Zhao-Lab-UW-DHO/AdaptFM.git
cd AdaptFM
chmod +x install.sh launch.sh
bash install.sh
```
Then, from the main AdaptFM folder (), start the program by running `bash launch.sh` 

Once AdaptFM is installed and working, check out the [User Guide](../user-guide/user-guide.md).