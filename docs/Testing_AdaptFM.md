# AdaptFM Demo Instructions

We have organized some datasets on [hugging face](https://huggingface.co/datasets/hbakhtiar/AdaptFM_Testing/tree/main) to test AdaptFM. The datasets are organized to follow the expected input structure for [standard AdaptFM usage](./Standard).

Once you have [installed](../README.md/#main-installation) the tool and downloaded the above datasets you can familiarize yourself with AdaptFM in the below workflow:

1. Follow our [testing](Standard/Testing-Models.md) instructions try out model inference with CellposeSAM and MicroSAM on organoid datasets, and SAM-Med-3D on radiologic data.
2. Once you have results you can [benchmark](Standard/Benchmarking-models.md) against the associated ground truth data
3. To fine-tune/train models, you can start by creating [annotations](Standard/Annotations.md), then [fine-tuning/training](Standard/Fine-tuning-models.md) the models.