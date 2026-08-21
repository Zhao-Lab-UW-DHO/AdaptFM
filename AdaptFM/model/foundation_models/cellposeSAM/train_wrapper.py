import argparse
import inspect
import json

from cellpose import core, io, models, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params")
    parser.add_argument("--train_dir")
    parser.add_argument("--test_dir")
    args = parser.parse_args()

    params = json.loads(args.params)
    params.pop("gpu")

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
