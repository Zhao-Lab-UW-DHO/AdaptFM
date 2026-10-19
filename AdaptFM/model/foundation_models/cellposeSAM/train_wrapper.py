import argparse
import inspect
import json
import torch

import torch.nn as nn
from cellpose import core, io, models, train

def unfreeze_last_n_modules(model: torch.nn.Module, n: int):
    # Step 1: Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Step 2: Unfreeze the last N child modules
    children = list(model.children())
    for child in children[-n:]:
        for param in child.parameters():
            param.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params")
    parser.add_argument("--train_dir")
    parser.add_argument("--test_dir")
    args = parser.parse_args()

    params = json.loads(args.params)
    params.pop("gpu")

    loss_name = params.pop("loss_fn", None)

    if loss_name:
        loss_class = getattr(nn, loss_name)

        # Replace the segmentation loss used by train_seg.
        train._loss_fn_seg = lambda lbl, y, device: loss_class()(
            y[:, -1],
            (lbl[:, -3] > 0.5).to(y.dtype),
        )

    sig = inspect.signature(train.train_seg)

    for key, value in params.items():
        if value == "None":
            params[key] = None
            continue

        if not isinstance(value, str):
            continue

        # Convert booleans
        if value.lower() in ["true", "false"]:
            params[key] = value.lower() == "true"
            continue

        # Infer type from the function's default value
        if key in sig.parameters:
            default = sig.parameters[key].default
            if default is not inspect.Parameter.empty and default is not None:
                try:
                    params[key] = type(default)(value)
                    continue
                except (ValueError, TypeError):
                    pass

        # Fallback: try int then float
        try:
            params[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            params[key] = float(value)
            continue
        except ValueError:
            pass

    io.logger_setup()  # run this to get printing of progress

    # Check if colab notebook instance has GPU access
    if core.use_gpu() == False:
        raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)
    print(args.train_dir)
    print(args.test_dir)

    output = io.load_train_test_data(args.train_dir, args.test_dir, mask_filter="_seg")
    train_data, train_labels, _, test_data, test_labels, _ = output

    kwargs = dict(params)

    # always overide these parameters - they are pulled from train_dir and test_dir that are user specified
    kwargs.pop("train_data", None)
    kwargs.pop("train_labels", None)
    kwargs.pop("test_data", None)
    kwargs.pop("test_labels", None)
    kwargs.pop("net", None)
    kwargs.pop("load_files", None)
    kwargs.pop("n_layers_to_unfreeze", None)

    if params["n_layers_to_unfreeze"] > 0:
        unfreeze_last_n_modules(model.net,n=params["n_layers_to_unfreeze"])

    if "nimg_per_epoch" in params:
        kwargs["nimg_per_epoch"] = max(2, len(train_data))

    print(f"train_data: {len(train_data)} images")
    print(f"train_labels: {len(train_labels)} labels")

    train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        **kwargs,
    )


if __name__ == "__main__":
    main()
