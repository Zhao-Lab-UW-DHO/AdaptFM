# MicroSAM

[Segment Anything for Microscopy](https://github.com/computational-cell-analytics/micro-sam) (MicroSAM), is a deep learning model designed to segment objects in microscopy images. It is based on the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). 

In AdaptFM, MicroSAM can be used in two ways:
    - **Inference** - use the trained MicroSAM model to segment images in bulk.
    - **Fine-tuning** - Adapt the existing MicroSAM model using your own labeled images so that it performs better on a specific cell type, tissue, imaging modality, or experimental system.

***Before using MicroSAM you must first install the environment and package with AdaptFM's environment manager***

## Using AdaptFM to Run Inference with MicroSAM

***Importantly, MicroSAM natively handles all image normalization and preprocessing. This is done by default, and normalizes pixels using a mix-max normalization strategy on a per z-slice basis to values between 0 and 255. The exact implementation can be found [here](https://github.com/computational-cell-analytics/micro-sam/blob/main/micro_sam/util.py#L618). Consequently, there is no need to independently normalize your data before running inference***

You can use MicroSAM in AdaptFM to run inference on images in bulk using the following steps:
 
1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference". Select "MicroSAM" as the model
2. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment.
3. Under "Output Directory" select the folder where you want to output your final predictions
4. Using "GPU Index" select the GPU you would like to use for running inference
5. Under "Model Checkpoint" navigate to the checkpoint you would like to use ***(optional)***. If you would like to use MicroSAM 'out-of-the-box' do not select a model checkpoint
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.

## Using AdaptFM to Fine-Tune MicroSAM

### Prepare the Data

***MicroSAM expects all image inputs for training to be values between 0 and 255 and in uint8 format. AdaptFM employs MicroSAM's expected normalization strategy of min-max normalization followed by setting values between 0 and 255.***

As with other AdaptFM models, you must prepare your data with the below steps before using the model:

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.

### Fine-Tuning the Model

1. Within AdaptFM navigate to the "Models" menu at the top and select "Training"
2. Select the model MicroSAM
3. In the "Dataset" field select "Browse" and navigate to the folder containing your raw and labeled images. This is the same folder in step 2 from "Prepare your data"
4. In the "Output Directory" field select "Browse" and navigate to the location where you want to output your results. For MicroSAM this will create two folders.
    - "training" - this will contain your raw images used for training
    - "segmentations" - this will contain the labels associated with the raw images used in training.
5. Using the "GPU Index" field, select the GPU you plan to use for training
6. **Optionally** you can select "Load Parameters" to further define MicroSAM fine-tuning. If you do not have programming and/or machine-learning experience, we do not recommend loading parameters or changing from any of the defaults. MicroSAM has many adjustable parameters defined by the authors. Full descriptions can be found [here](https://github.com/computational-cell-analytics/micro-sam/blob/main/micro_sam/training/training.py#L228). We have provided descriptions of some of the most common parameters here:
    - n_epochs - how many times the entire training set is evaluated during training. If training is taking a particularly long time, try decreasing this number. Decreasing this too much can limit model performance
    - n_objects_per_batch - the number of objects required to compute a batch. If your images are very sparse, try decreasing this value. The default value is 25
    - checkpoint_path - the path to the checkpoint used for training
    - save_root - optional directory to save checkpoints and logs
    - loss_fn - micro_sam does not natively support different loss functions. ***Use the below steps to use a different loss function***


#### Adjusting The Loss Function

MicroSAM by default uses a DiceBasedDistanceLoss function described [here](https://constantinpape.github.io/torch-em/torch_em/loss/distance_based.html). MicroSAM's primary training function does not have an adjustable parameter for the loss function. To add a different loss function, you follow the below steps:

1. Navigate to the "micro-sam_adapt" conda environment and open the micro_sam training code under micro-sam_adapt > lib > python3.10 > site-packages > micro_sam > training > training.py 
2. Under the train_sam function, add a parameter called 'loss_fn'

```python

def train_sam(
    name: str,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    early_stopping: Optional[int] = 10,
    n_objects_per_batch: Optional[int] = 25,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    with_segmentation_decoder: bool = True,
    freeze: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    lr: float = 1e-5,
    n_sub_iteration: int = 8,
    save_root: Optional[Union[str, os.PathLike]] = None,
    mask_prob: float = 0.5,
    n_iterations: Optional[int] = None,
    scheduler_class: Optional[_LRScheduler] = torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    save_every_kth_epoch: Optional[int] = None,
    pbar_signals: Optional[QObject] = None,
    optimizer_class: Optional[Optimizer] = torch.optim.AdamW,
    peft_kwargs: Optional[Dict] = None,
    ignore_warnings: bool = True,
    verify_n_labels_in_loader: Optional[int] = 50,
    box_distortion_factor: Optional[float] = 0.025,
    overwrite_training: bool = True,
    strict_decoder_loading: bool = True,
    loss_fn: str = "DiceBasedDistanceLoss" # <-- ADD THIS HERE
    **model_kwargs,
) -> None:

```

3. At [line 320](https://github.com/computational-cell-analytics/micro-sam/blob/main/micro_sam/training/training.py#L320) add the following:

```python
def get_loss(loss_name, **kwargs):
    loss_cls = getattr(torch_em.loss, loss_name)
    return loss_cls(**kwargs)

instance_seg_loss = get_loss(
    loss_fn,
    mask_distances_in_bg=True
)
```

and at line 322, remove:

```python
instance_seg_loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)
```
