# Benchmarking Models Using AdaptFM

You can use AdaptFM to determine which model performed best on a specific task. Steps are:
1. Select "Benchmark" --> "Run Benchmark".
2. Decide the Metric you would like to measure, optionally you can add your own metric.
3. "Select Ground Truth" --> the folder with your ground truth images. These should be named **identically** to your model predictions.
4. "Add model predictions" --> select the folder with the output images that you selected when [testing](./Testing-Models.md).
5. If your models aren't performing well on benchmarks, consider [fine-tuning the models](./Fine-tuning-models.md).