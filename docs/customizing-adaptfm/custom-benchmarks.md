# Adding Benchmarks to AdaptFM

In addition to the existing segmentation benchmarks, AdaptFM also allows users to add new benchmarking algorithms. The steps are:

1. Navigate to AdaptFM > gui > widgets > metrics_widget.py
2. Create a new class that inherits from the "Metric" base class. Register it with MetricRegistry
3. Provide your class a 'name' and place your benchmarking algorithm in a 'compute' method. This method should take a parameter for 'ground_truth' files and 'model_dirs' (folders containing predictions) and 'output_dir' (where you would like to save your results).
4. The benchmarking algorithm should now appear in AdaptFM's benchmark module

Example:

```python
@MetricRegistry.register
class DiceScore(Metric):
    name = "Dice Score" # <-Define a name to appear in the benchmark widget

    def compute(self, gt_dir, models_dirs, output_dir): # < - define what the metric does. Include parameters for the ground truth, model predictions, and output directory
        gt_dir = Path(gt_dir)
        gt_files = glob_masks(gt_dir)
        results = {}

        for display_name, model_dir in models_dirs.items():
            model_dir = Path(model_dir)
            model_files = glob_masks(model_dir)
            file_pairs = list(zip(gt_files, model_files))

            per_image_dice = run_parallel(
                file_pairs,
                self._dice_worker,
                self.n_processes,
            )
            results[display_name] = per_image_dice

        plot_dice_boxplot(results, gt_dir, self.name, output_dir) # <-- Plot results in a boxplot, or write you own custom plot function
        return results

    @staticmethod
    def _dice_worker(pair):
        gt_file, pred_file = pair

        gt = read_mask(gt_file) > 0
        pred = read_mask(pred_file) > 0

        intersection = (gt & pred).sum()
        union = gt.sum() + pred.sum()

        return 2 * intersection / union if union > 0 else 1.0
```