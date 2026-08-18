# SSVT

SSVT is a custom vision transformer model pretrained on ~180,000 3D organoid confocal microscopy images. The checkpoint is publicly available for download on [hugging face](https://huggingface.co/hbakhtiar/SSVT_Organoids/tree/main). SSVT is supported directly within AdaptFM. Importantly, SSVT is not designed to perform segmentation out-of-the-box. You **must** first fine-tune SSVT for a segmentation task, then use it for downstream inference. 

***Before using SSVT you must first download the checkpoint from huggingface with AdaptFM's environment manager***

## Training an nnU-Net model in AdaptFM

### Prepare the data

To train a model with SSVT in AdaptFM you must first prepare the data with the below steps:

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.

### Train the Model

1. Within AdaptFM navigate to the "Models" menu at the top and select "Training"
2. Select the model SSVT
3. In the "Dataset" field select "Browse" and navigate to the folder containing your raw and labeled images. This is the same folder in step 2 from "Prepare your data"
4. In the "Output Directory" field select "Browse and navigate to the location where you want to output your results. 
5. Using the "GPU Index" field, select the GPU you plan to use for training
6. **Required** select "Load Parameters" to define fine-tuning for SSVT. 
    - CKPT_path - the file path to the SSVT checkpoint
    - samples_per_volume - the number of objects required in each volume to be used for training. Default is 48. If your volume's are very sparse, consider lowering this.
    - batch_size - Number of 3D image samples processed together before the model updates its weights. The default is 8. If you get out of memory errors while training, consider lowering this.
    - num_workers -  Number of CPU workers used to load and prepare training data. The default is 16
    - lr - learning rate. Controls how strongly the model's weights are adjusted during each training update. Default is 1e-5
    - weight_decay - Regularization that discourages overly complex model weights and can help reduce overfitting. Default is 1e-4
    - unfreeze_epoch - After how many epochs should the encoder be unfrozen. By default only the decoder weights are udpated. The default unfreeze_epoch is 100
    - total_epochs - number of complete passes through the training data
7. Once you are ready you can select "Run" to begin training the model. You can monitor output in the "Output Log" section. If at any point you would like to stop training simply click the "Terminate" button.

## Using an SSVT model for Inference in AdaptFM

You can use SSVT in AdaptFM to run inference on images in bulk using the following steps:
 
1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference". Select "SSVT" as the model. 
2. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment.
3. Under "Output Directory" select the folder where you want to output your final predictions
4. Using "GPU Index" select the GPU you would like to use for running inference
5. Under "Model Checkpoint" navigate to the SSVT checkpoint you trained.
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.
