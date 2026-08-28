# Model Leaderboard

The best models for your application will depend on your specific images and segmentation task. ***We recommend starting by benchmarking several models on a small, representative subset of your data to test performance. If needed, fine-tune all models, as zero-short performance does not always predict fine-tuned performance.***

The results below provide additional guidance by showing how the supported models performed across several representative datasets and segmentation tasks evaluated by the AdaptFM team. 

### Broad Synthetic Leukemia Cell Line

Below is performance of the cell segmentation models on Broad leukemia cell lines:

| Model | Average Dice Score |
| --- | --- |
| nnU-Net (trained) | 0.87 |
| SSVT (fine-tuned) | 0.77 |
| MicroSAM (zero-shot) | 0.75 |
| CellposeSAM (zero-shot) | 0.81 | 
| CellposeSAM (fine-tuned) | 0.92 |
| MicroSAM (fine-tuned) | 0.82 | 
| CellSAM (zero-shot) | 0.78 |

### Organoid Total and Dead Nuclear Counts

We evaluated the performance of the microscopy segmentation models on roughly ~3,700 manually counted 3D organoids from confocal microscopy images. Below are the Pearson R values of the predicted model counts to the human counts, along with the slope of the line of best fit

| Model | Pearson R | Slope - line of best fit|
| --- | --- | --- |
| nnU-Net (trained) | 0.81 | 0.86 | 
| SSVT (fine-tuned) | 0.82 | 0.761 | 
| MicroSAM (zero-shot) | 0.81 | 0.76 | 
| CellposeSAM (zero-shot) | 0.82 | 0.69 |
| MicroSAM (fine-tuned) | 0.77 | 0.85 | 
| CellposeSAM (fine-tuned) | 0.82 | 1.1 |
| CellSAM (zero-shot) | 0.77 | 0.57 |

Using the same image sets, we evaluated how well the models enumerated nuclei with a death marker, a significantly harder task than a traditional nuclear stain. Below are the Pearson R values of the predicted model counts to the human counts, along with the slope of the line of best fit

| Model | Pearson R | Slope - line of best fit|
| --- | --- | --- |
| nnU-Net (trained) | 0.54 | 0.57 | 
| SSVT (fine-tuned) | 0.62 | 0.79 | 
| MicroSAM (zero-shot) | 0.56 | 0.97 | 
| CellposeSAM (zero-shot) | 0.48 | 1.08 |
| MicroSAM (fine-tuned) | 0.64 | 0.79 | 
| CellposeSAM (fine-tuned) | 0.63 | 0.58 |
| CellSAM (zero-shot) | 0.72 | 0.68 |

### Performance on imageTBAD aorta segmentation

Using the imageTBAD dataset, we compared performance of CTFM and SAM-Med3D in segmenting the aorta. We excluded Merlin-nnUNet as it is not trained to identify the aorta.

| Model | Average Dice Score |
| --- | --- |
| SAM-Med3D (zero-shot) | 0.60 |
| CT-FM (zero-shot) | 0.26 |

