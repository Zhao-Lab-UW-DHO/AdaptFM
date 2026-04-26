# AdaptFM Demo Instructions

We have organized some datasets on [hugging face](https://huggingface.co/datasets/hbakhtiar/AdaptFM_Testing/tree/main) to test AdaptFM. The datasets are organized to follow the expected input structure for [standard AdaptFM usage](./Standard).

Once you have [installed](../README.md/#main-installation) the tool (including the installation of models that you will be using in the demo) and downloaded the above datasets you can familiarize yourself with AdaptFM in the below workflow:

1. Follow our [testing](Standard/Testing-Models.md) instructions try out model inference with CellposeSAM and MicroSAM on organoid datasets (BBBC024, BBBC027, or Organoids_Nuclear_Channel), and SAM-Med-3D on radiologic data (RESECT or imageTBAD).
1. Once you have results you can [benchmark](Standard/Benchmarking-models.md) against the associated ground truth data.
- Note that if doing a counts comparison on the Organoids_Nuclear_Channel, you should select nuclear_true_counts.csv file.
1. Create [annotations](Standard/Annotations.md) using the annotation widget. The annotation tool is intended to generate the labels for the next step using the Save for training button.
    - If you prefer to skip manual annotation of enough labels for fine-tuning, you can use the Organoids_Nuclear_Channel dataset by placing a subset of the matched labels and raw images into a single folder to select for training.
1. [Fine-tune/train](Standard/Fine-tuning-models.md) the models using the data from the previous step.
