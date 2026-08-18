# Quick Start Guide

Welcome to AdaptFM! This guide covers the fastest way to install the base framework and launch the GUI. 

> **Note:** If you are installing AdaptFM on a custom hardware setup (e.g. with an AMD GPU) or want to use Apptainer/Singularity containers, please skip this page and follow our [Extended Installation](install-details.md).

---

## Prerequisites

To use our automated installation script (`install.sh`), your system must be running Linux and have [Conda already installed](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html). Furthermore, the automated installer relies on having an NVIDIA GPU installed and working (with nvidia-smi available in the command line interface). If you try the install and your machine does not meet these requirements try the documentation for [Extended Installation Details](install-details.md).

AdaptFM should be installable on any system capable of running a modern version of pytorch (with GPU integration for practical reasons) it is tested to run on the default image for Windows Subsystem for Linux 2 (WSL2) with an enterprise NVIDIA GPU. With a manual installation, we have found AdaptFM to work on Arch Linux: 6.18-lts kernel using Wayland and an AMD Radeon RX 9070 XT GPU, and on Windows 11 natively without GPU integration (not recommended)

AdaptFM is extensively tested on Ubuntu LTS with enterprise NVIDIA GPUs.

GPU Driver versions with CUDA >= 12.6 are supported as older versions are not supported in external code repositories integrated into AdaptFM. 

---

## 1. Installation

The easiest way to install AdaptFM is with our automated installer in the command line interface of a Linux computer system. Windows 11 Users can install and run [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) within Windows for a Linux system that works with the installer.

**Clone the repository and install**
```bash
git clone https://github.com/Zhao-Lab-UW-DHO/AdaptFM.git
cd AdaptFM
chmod +x install.sh launch.sh
bash install.sh
```
then launch with `bash launch.sh`