# Adding Benchmarks to AdaptFM

In addition to the existing segmentation benchmarks, AdaptFM also allows users to add new benchmarking algorithms. The steps are:

1. Navigate to AdaptFM > gui > widgets > metrics_widget.py
2. Create a new class that inherits from the "Metric" base class. Register it with MetricRegistry
3. Provide your class a 'name' and place your benchmarking algorithm in a 'compute' method. This method should take a parameter for 'ground_truth' files and 'prediction' files.
4. The benchmarking algorithm should now appear in AdaptFM's benchmark module

Example:

```python


@MetricRegistry.register
class DiceScore(Metric):
    name = "Dice Score"

    def compute(self, gt_dir, models_dirs):
        """
        gt_dir: path to ground truth folder
        models_dirs: list of paths to model prediction folders
        """
        gt_dir = Path(gt_dir)
        results = {}

        # get all ground truth files
        gt_files = sorted(gt_dir.glob("*"))  # assumes all images in folder

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*"))

            per_image_dice = []

            # match files by order (or implement matching by name if needed)
            for gt_file, pred_file in zip(gt_files, model_files):
                gt = tiff.imread(gt_file) > 0        # binarize
                pred = tiff.imread(pred_file) > 0    # binarize
                intersection = (gt & pred).sum()
                union = gt.sum() + pred.sum()
                dice = 2 * intersection / union if union > 0 else 1.0
                per_image_dice.append(dice)

            # store results for this model
            results[str(model_dir)] = per_image_dice

        return results
