#!/bin/bash
set -e
cd "$(dirname "$(readlink -f "$0")")"

IMAGE="AdaptFM.sif"

if [ ! -f "$IMAGE" ]; then
    # possible env vars that fix errors: APPTAINER_TMPDIR, APPTAINER_CACHEDIR
    apptainer build --fakeroot "$IMAGE" AdaptFM_apptainer.def
fi


# May need to bind extra items in case of issues e.g.
# if machine-id is said to be missing/invalid/no permission
# try --bind /etc/machine-id:/etc/machine-id
apptainer run \
    --nv \
    --bind /mnt:/mnt \
    --bind $HOME:$HOME \
    --bind $XAUTHORITY:/tmp/.Xauthority-container \
    --env XAUTHORITY=/tmp/.Xauthority-container \
    --env DISPLAY=$DISPLAY \
    "$IMAGE"