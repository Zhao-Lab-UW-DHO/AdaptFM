cd "$(dirname "$(readlink -f "$0")")/../"
set -e
eval "$($(command -v conda) 'shell.bash' 'hook')"

### Block for AdaptFM base repo tests
echo "RUNNING AdaptFM tests"

conda activate "AdaptFM"
# test 2d seg methods
python -m unittest tests/test_py_segmentations.py

echo "FINISHED AdaptFM tests"
### END Block for AdaptFM base repo tests

### Block for MicroSAM tests
echo "RUNNING MicroSAM tests"

conda activate "micro-sam_adapt"
# pip install simpleitk --no-deps # importing the model registry has this dependency. there are no other deps
# test 2d seg methods
python -m unittest tests/test_microsam.py

echo "FINISHED MicroSAM tests"
### END Block for AdaptFM base repo tests