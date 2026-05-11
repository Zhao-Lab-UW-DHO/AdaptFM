# Benchmarking Models Using AdaptFM

You can use AdaptFM to determine which model performed best on a specific task. Steps are:
1. Select "Benchmark" --> "Run Benchmark".
2. Decide the Metric you would like to measure, optionally you can add your own metric.
3. "Select Ground Truth" --> the folder with your ground truth images. These should be named **identically** to your model predictions.
- If performing a counts comparison, you shoudld first create a csv file with two columns: 1.) File names that identically match the names of your image files. 2.) Counts with the ground-truth object counts. See Organoids_Nuclear_Channel's ground-truth csv as an example [link](https://huggingface.co/datasets/hbakhtiar/AdaptFM_Testing/tree/main)
5. "Add model predictions" --> select the folder with the output images that you selected when [testing](./Testing-Models.md).
6. If your models aren't performing well on benchmarks, consider [fine-tuning the models](./Fine-tuning-models.md).
7. All metrics except for counts comparison can be made faster using multiple processes.
8. Benchmark plots are saved to the folder containing ground truth images

Example benchmark plots for [counts comparison](../../asset/Compare_counts_example.jpg) and [boxplots](../../asset/Example Dice Boxplot.jpg)
