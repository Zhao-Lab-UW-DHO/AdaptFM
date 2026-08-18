#!/bin/bash
set -e
cd "$(dirname "$(readlink -f "$0")")"

eval "$(conda shell.bash hook)"
conda activate AdaptFM

QT_API="pyqt6" python -m AdaptFM.dev_launch