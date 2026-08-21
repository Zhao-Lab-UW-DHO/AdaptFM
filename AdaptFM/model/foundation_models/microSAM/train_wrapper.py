# train_wrapper.py
import argparse
import json

from micro_sam.training import default_sam_loader, train_sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_paths")
    parser.add_argument("--label_paths")
    parser.add_argument("--params")
    parser.add_argument("--out")
    args = parser.parse_args()

    raw_paths = json.loads(args.raw_paths)
    label_paths = json.loads(args.label_paths)
    params = json.loads(args.params)

    # Build dataloaders INSIDE the MicroSAM environment
    train_loader = default_sam_loader(
        raw_paths=raw_paths,
        label_paths=label_paths,
        raw_key=None,
        label_key=None,
        with_segmentation_decoder=params["with_segmentation_decoder"],
        patch_shape=(512, 512),
        batch_size=1,
        is_seg_dataset=True,
        shuffle=True,
        raw_transform=None,
        sampler=None,
    )

    val_loader = default_sam_loader(
        raw_paths=raw_paths,
        label_paths=label_paths,
        raw_key=None,
        label_key=None,
        with_segmentation_decoder=params["with_segmentation_decoder"],
        patch_shape=(512, 512),
        batch_size=1,
        is_seg_dataset=True,
        shuffle=False,
        raw_transform=None,
        sampler=None,
    )

    def as_int(x):
        return None if x is None else int(x)

    def as_bool(x):
        if isinstance(x, bool):
            return x
        return str(x).lower() in ("1", "true", "yes", "y")

    if params["model_type"] == "<class 'inspect._empty'>":
        print("no model_type provided, defaulting to vit_b_lm")
        params["model_type"] = "vit_b_lm"

    train_sam(
        name=str(params["name"]),
        save_root=args.out,
        model_type=str(params["model_type"]),
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=as_int(params["n_epochs"]),
        n_objects_per_batch=as_int(params["n_objects_per_batch"]),
        with_segmentation_decoder=as_bool(params["with_segmentation_decoder"]),
        device="cuda",
    )


if __name__ == "__main__":
    main()
