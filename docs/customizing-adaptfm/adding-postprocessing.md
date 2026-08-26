# Adding New Postprocessing Algorithms to AdaptFM

AdaptFM supports customization/addition of postprocessing algorithms. These algorithms are primarily meant for segmented images (images that have already been segmentation using one of AdaptFM's segmentation models). The steps to adding a new algorithm are:

1. Navigate to AdaptFM > postprocessing > pipelines and create a new file with your postprocessing algorithm
2. Navigate to AdaptFM > postprocessing > post_proc_spec.py and write a new class that inherits from the PostProcess base class
3. Use the run_postprocess method to perform any data preprocessing prior to launching the postprocessing script. Be sure the launch the postprocessing pipeline in a subprocess so it doesn't hang on the main AdaptFM.

```python
class USegment3DSpec(PostProcess):
    def __init__(self, name, conda_env, module_path):
        super().__init__(name, conda_env, module_path)
        self.name = name
        self.conda_env = conda_env
        self.module_path = module_path

    def run_postprocess(self, input_dir, output_dir, gpu): #<- The main function that launches your postprocessing pipeline

        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        post_process_cmd = [
            "python",
            "-m",
            f"{self.module_path}",
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
        ]

        cmd = self._wrap_with_conda(post_process_cmd)

        subprocess.Popen( # <- launch as a subprocess
            cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )

```

4. Navigate to AdaptFM > postprocessing > post_proc_registry.py, import your new class at the top, and add it to the POST_PROC_REGISTRY. If the pipeline needs to run in a separate conda environment, write the path to the conda environment in 'conda_env'. 
    - Be sure to set the module_path to the path to the pipeline 

```python

from AdaptFM.postprocessing.post_proc_spec import USegment3DSpec # <- import here

POSTPROC_REGISTRY = {
    "USegment3D": USegment3DSpec(
        name="USegment3D",
        conda_env=_read_prefix("usegment3d_adapt"), # add conda env if needed
        module_path=str(ADAPTFM_POSTPROC_PATH / "pipelines" / "useg3d.py"), # add path to the pipeline
    )
}
```