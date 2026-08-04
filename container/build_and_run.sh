cd "$(dirname "$(readlink -f "$0")")"

apptainer build --fakeroot AdaptFM.sif AdaptFM_apptainer.def