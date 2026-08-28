# CT-FM

[CT-FM](https://github.com/project-lighter/CT-FM) is a 3D image-based pretrained foundation model for a number of radiological tasks. It was pretrained using 148,000 CT scans. By default CT-FM comes with two base models that are adapted in AdaptFM:

1. Whole_body_Segmentation model - This model can be used out-of-the-box to segment 118 different anatomical structures in CT scans. 
2. CT-FM SegRestNet - this takes the pretrained CT-FM model, and allows users to fine-tune their own segmentation model using CT-FM (not the whole body segmentation model) as a starting point

Within AdaptFM, CT-FM's models can be used in two ways
    -**Inference** - Run inference using CT-FM's whole_body_segmentation model
    -**Fine-Tuning** - Fine-tune a new segmentation model based on the individual structure you want to identify. This is based off of CT-FM's pretrained base model. 

***Before using CT-FM you must first install the environment and package with AdaptFM's environment manager***

## Using AdaptFM to Run Inference with CT-FM

***CT-FM natively handles all normalization and preprocessing. This is done by a two step process in the pipeline in AdaptFM. You do not need to reimplement this.***
1. Intensities outside of (-1024,2048) are clipped 
2. The clipped values then undergo mix-max normalization to scale them between 0 and 1

You can use CT-FM in AdaptFM to run inference on images in bulk using the following steps:
 
1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference". Select "MicroSAM" as the model
2. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment.
3. Under "Output Directory" select the folder where you want to output your final predictions
4. **GPU Index** is not used in CT-FM by default for inference. CT-FM consumes a very large amount of memory during inference and can lead to crashes. Consequently, all CT-FM inference is run on a CPU.
5. Under "Model Checkpoint" navigate to the checkpoint you would like to use. If you don't select a model, the whole_body_segmentation model will be used by default. 
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.

## Using AdaptFM to Fine-Tune CT-FM

### Prepare the Data

***CT-FM normalizes data using the above two step process (clipping values, followed by min-max normalization) This is employed and handled by the model itself***

As with other AdaptFM models, you must prepare your data with the below steps before using the model:

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.

### Fine-Tuning the Model

1. Within AdaptFM navigate to the "Models" menu at the top and select "Training"
2. Select the model CT-FM
3. In the "Dataset" field select "Browse" and navigate to the folder containing your raw and labeled images. This is the same folder in step 2 from "Prepare your data"
4. In the "Output Directory" field select "Browse" and navigate to the location where you want to output your results. 
5. Using the "GPU Index" field, select the GPU you plan to use for training. 
6. You can **optionally** select "Load Parameters" to further define the training parameters. 
    - batch_size - Number of 3D image samples processed together before the model updates its weights. The default is 2. If you get out of memory errors while training, consider lowering this.
    - max_epochs - number of complete passes through the training data. Default is 300
    - learning_rate - Controls how strongly the model's weights are adjusted during each training update. Default is 2e-4
    - num_workers - Number of CPU workers used to load and prepare training data. The default is 16
    - cache_dir - temporary directory used during training
7. Once you are ready you can select "Run" to begin training the model. You can monitor output in the "Output Log" section. If at any point you would like to stop training simply click the "Terminate" button.
