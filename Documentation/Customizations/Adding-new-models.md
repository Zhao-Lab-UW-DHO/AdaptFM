# Adding New Models to AdaptFM

Users can add new models for inference or training. There are four main steps outlined below.

### Creating a New Conda Environment

To avoid version conflicts, AdaptFM requires each model's repo be compiled in a separate conda environment. 

Follow the instructions on the model's page for installing it. 

### Defining Training

There are two parts to defining training: Creating a new model specification class and writing a wrapper. 

1. Creating a new model specification class

- Navigate to AdaptFM > model > fmSpec.py and create a new class that inherits from FoundationModelSpec
- Define a prepare_dataset method that preprocesses your data. Do any normalization required by your model here. Example from CellposeSAMSpec

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
asdfasdfasdf

