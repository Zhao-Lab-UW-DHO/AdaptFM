#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")" # Change script working directory to the AdaptFM repo, considered cd "${0%/*}" but returns err when already in working directory.
echo "======================================================="
echo " AdaptFM Install Script"
echo "======================================================="

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: 'nvidia-smi' utility could not be executed from shell. NVIDIA drivers are required for AdaptFM."
    echo "Could not install AdaptFM"
    exit 1
fi
NVIDIA_SMI_OUT=$(nvidia-smi)
CUDA_VERSION=$(echo "$NVIDIA_SMI_OUT" | grep -oP "(?:CUDA Version|CUDA UMD Version):\K\s*[0-9.]*" | xargs)
CUDA_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f2)

if ! command -v conda &> /dev/null; then
    echo "ERROR: 'conda' could not be executed from the shell."
    echo "Please install Conda or ensure it is present in your shell's PATH environment variable."
    echo "Could not install AdaptFM"
    exit 1
fi
eval "$(conda shell.bash hook)"
conda create -n AdaptFM python=3.12 -y
conda activate AdaptFM

case ${CUDA_MAJOR} in
    12)
        if [ "${CUDA_MINOR}" -lt 6 ]; then
            echo "ERROR: CUDA version 12.${CUDA_MINOR} is below the supported 12.6 minimum."
            echo "Could not install AdaptFM"
            exit 1
        fi
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
        CONDA_CUDAVER="12.6"
        ;;
    13)
        if [ "${CUDA_MINOR}" -lt 2 ]; then
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
            CONDA_CUDAVER="13.0"
        else
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu132"
            CONDA_CUDAVER="13.2"
        fi
        ;;
    *)
        echo "ERROR: Detected CUDA Version (${CUDA_VERSION}) falls outside explicitly supported ranges (12.6 - 13.x)."
        echo "Could not install AdaptFM"
        exit 1
        ;;
esac

# At time of writing the pip install order determines if setuptools (repo then torch) 
# or numpy (torch then repo) reinstalls to a different version
pip install -e .
pip install torch torchvision --index-url "${PYTORCH_INDEX}"
conda install cuda-toolkit cuda-nvcc "cuda-version=${CONDA_CUDAVER}" ninja -c conda-forge -y

echo "======================================================="
echo "AdaptFM was sucessfully installed"
echo "======================================================="