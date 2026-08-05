# Adding New Models to AdaptFM

Users can add new models for inference or training. There are two main steps outlined below.

### Creating a New Conda Environment

To avoid version conflicts, AdaptFM requires each model's repo be installed in a separate conda environment. 

Follow the instructions on the model's page for installing it. 

### Defining a new model

There are two parts to defining training: Creating a new model specification class and writing an API. 

1. Creating a new model specification class

- Navigate to AdaptFM > model > fmSpec.py and create a new class that inherits from FoundationModelSpec
- Define a prepare_dataset method that preprocesses your data. Do any normalization required by your model here. It should return a dictionary with specific parameters you intend to use for training. Example from CellposeSAMSpec

```python
    def prepare_dataset(self, dataset_manager, output_dir,params):

        import random
        import tifffile as tiff
        output_dir = Path(output_dir)

        training_dir = output_dir / "training"
        testing_dir  = output_dir / "testing"

        training_dir.mkdir(parents=True, exist_ok=True)
        testing_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # 1. Train / test split
        # -------------------------
        samples = dataset_manager.samples.copy()
        random.shuffle(samples)

        split_idx = int(0.8 * len(samples))
        train_samples = samples[:split_idx]
        test_samples  = samples[split_idx:]

        # -------------------------
        # 2. Helper to write slices
        # -------------------------
        def write_slices(samples, out_dir):
            for s in samples:
                img = tiff.imread(s["image"])    # shape: (z, y, x)
                mask = tiff.imread(s["mask"])    # same shape

                base_name = Path(s["image"]).stem  # no suffix

                for z in range(img.shape[0]):
                    img_out  = out_dir / f"{base_name}_z{z}.tiff"
                    mask_out = out_dir / f"{base_name}_z{z}_seg.tiff"

                    tiff.imwrite(img_out, img[z], compression="zlib")
                    tiff.imwrite(mask_out, mask[z], compression="zlib")

        # -------------------------
        # 3. Write datasets
        # -------------------------
        write_slices(train_samples, training_dir)
        write_slices(test_samples, testing_dir)

        return {
            "train_dir": training_dir,
            "test_dir": testing_dir,
        }
```

- Now create a training command that specifies anything returned by your prepare_dataset dictionary. Example:

```python
    def training_command(self, dataset_info, params, run_dir):
        """
        CellposeSAM training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """
        return [
            "python",
            f"{self.training_wrapper_path}",
            "--train_dir", str(dataset_info["train_dir"]),
            "--test_dir", str(dataset_info["test_dir"]),
            "--params", json.dumps(params),
        ]

```

- Navigate to AdaptFM > model > foundation_models - create a new folder for your model and add train_wrapper.py and inference_wrapper.py files (see AdaptFM > model > foundation_models > cellposeSAM > train_wrapper.py as example for what to include in these scripts). In short, it should call the training or inference function used by your new model.
- Navigate to AdaptFM > model > registry.py and add your model to the new registry, specifying the path to the conda environment, and train_wrapper.py/inference_wrapper.py files

```python
    "CellposeSAM": CellposeSAMSpec(
        name="CellposeSAM",
        conda_env="/path/to/conda_envs/cellpose",
        module_path="cellpose.train",
        training_wrapper_path = str(ADAPTFM_MODEL_PATH / "foundation_models" / "cellposeSAM" / "train_wrapper.py"),
        inference_wrapper_path = str(ADAPTFM_MODEL_PATH / "foundation_models" / "cellposeSAM" / "inference_wrapper.py"),
    ),
```
