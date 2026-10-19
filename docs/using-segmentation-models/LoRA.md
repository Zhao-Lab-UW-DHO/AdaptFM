# Low Rank Adaptation (LoRA)

Low rank adaptation (LoRA) is a parameter-efficient fine-tuning technique that lets users adapt large foundation models without needing to retrain the entire model. We have incorporated LoRA as an option within AdaptFM. You can use LoRA to fine-tune **any** of the available models within AdaptFM.

***Before using LoRA, you must first install the environment and package for the model you would like to use. For example, if using LoRA with CellSAM, first use AdaptFM's environment installer to install CellSAM***

## Fine-tuning With LoRA

### Prepare the Data

To fine-tune a model using LoRA in AdaptFM, you must prepare your data with the below steps before using the model:

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.


### Train the Model

Prior to fine-tuning your model, you must first install the environment using the AdaptFM model installer

1. Within AdaptFM navigate to the "Segmentation Models" menu at the top and select "Training"
2. Select "LoRA" in the model dropdown
3. In the "Dataset" field select "Browse" and navigate to the folder containing your raw and labeled images. This is the same folder in step 2 from "Prepare your data"
4. In the "Output Directory" field select "Browse" and navigate to the location where you want to output your results. 
5. Using the "GPU Index" field, select the GPU you plan to use for training
6. **Required** select "Load Parameters" to define fine-tuning for LoRA. If you do not have a machine learning background you can start with the default parameters (except for ckpt_path and model_env)
    - ckpt_path- the full path for the model checkpoint. This field is only required for the following models:
        - BME-X
        - SSVT
        - SAMMed3D
        - nnUNet
        - Merlin nnUNet
    ***For nnUNet and Merlin nnUNet enter the path to the 'nnUNetTrainer__nnUNetPlans__3d_fullres' folder in the nnUNet_results folder***
    - target_modules - substring filter that controls which specific layers inside the model get replaced with LoRA adapters
    - lora_r - LoRA rank dimension (r); determines the bottlenck rank and total trainable parameter count. Default is 8
    - lora_alpha - scaling factor that controls the weight magnitude of LoRA updates relative to the rank. Default is 16
    - lora_dropout - dropout probability applied to the input of the LoRA adapter layers duribng training. Default is 0
    - epochs - total number of complete passes through the training dataset. Default is 500
    - patch_size - spatial or volumetric dimensions used for cropping input images
    - batch_size - number of image/volume samples processed concurrettly in a single foward and backward pass. Default is 1.
    - num_workers - number of CPUs used for preprocessing data during training. Default is 1
    - learning_rate - step size used during optimizer to update trainable LoRA adapter parameters
    - weight_decay - L2 regularization penalty applied to adapter weights
    - model_env - the name of the model you are fine-tuning. The name **must** match the installed environment name. The options are:
        - cellpose_adapt
        - micro-sam_adapt
        - cellsam_adapt
        - AdaptFM (if using SSVT)
        - CTFM_adapt
        - BME-X_adapt
        - nnUNet_adapt
        - Merlin_nnUNet_adapt
        - sammed3d_adapt

7. Once you are ready you can select "Run" to begin training the model. You can monitor output in the "Output Log" section. If at any point you would like to stop training simply click the "Terminate" button.

## Using a LoRA Fine-Tuned Model for Inference in AdaptFM

You can use LoRA fine-tuned models in AdaptFM to run inference on images in bulk using the following steps:
 
1. Within AdaptFM navigate to the "Segmentation Models" menu at the top and select "Inference". Select "LoRA" as the model. 
2. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment.
3. Under "Output Directory" select the folder where you want to output your final predictions
4. Using "GPU Index" select the GPU you would like to use for running inference
5. Under "Model Checkpoint" navigate to the LoRA checkpoint you trained. This will be located in the output folder from your training step. **Be sure to select the 'adapter_model.pt' file'**
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.