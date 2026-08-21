import argparse
from pathlib import Path

import tifffile as tiff
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir")
    parser.add_argument("--output_path")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    run_microsam_inference(
        test_dir=args.dataset_dir,
        output_path=args.output_path,
        checkpoint=args.checkpoint,
    )


def run_microsam_inference(test_dir, output_path, checkpoint):

    if checkpoint == "None":
        checkpoint = None

    if checkpoint is not None:
        predictor, segmenter = get_predictor_and_segmenter(
            "vit_b_lm", device="cuda", checkpoint=checkpoint
        )

    else:
        predictor, segmenter = get_predictor_and_segmenter("vit_b_lm", device="cuda")

    images2test = [f.name for f in Path(test_dir).iterdir() if f.is_file()]

    for image_name in images2test:
        image_path = str(Path(test_dir) / image_name)
        image = tiff.imread(image_path)

        try:
            segmented_image = automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=image,
                verbose=False,
            )
        except Exception as e:
            print("Image", image_name, "encountered exception", e)
            continue

        Path(output_path).mkdir(parents=True, exist_ok=True)

        image_output_path = str(Path(output_path) / image_name)
        tiff.imwrite(image_output_path, segmented_image)


if __name__ == "__main__":
    main()
