from micro_sam.automatic_segmentation import get_predictor_and_segmenter,automatic_instance_segmentation
import argparse
import os
import tifffile as tiff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir")
    parser.add_argument('--output_path')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    test_dir = args.dataset_dir
    output_path = args.output_path
    checkpoint = args.checkpoint
    
    if checkpoint is not None:
        
        predictor, segmenter = get_predictor_and_segmenter('vit_b_lm',
                                                        device='cuda',
                                                        checkpoint=checkpoint)

    else:
         
        predictor, segmenter = get_predictor_and_segmenter('vit_b_lm',
                                                        device='cuda')
    
    images2test = os.listdir(test_dir)

    for image_name in images2test:
        image_path = os.path.join(test_dir, image_name)
        image = tiff.imread(image_path)

        try:
        
                segmented_image =  automatic_instance_segmentation(predictor=predictor,
                                                segmenter=segmenter,
                                                input_path=image,
                                                verbose=False)
                

        except Exception as e:
                continue
        
        
        os.makedirs(output_path,exist_ok=True)

        image_output_path = os.path.join(output_path,image_name)
        tiff.imwrite(image_output_path,segmented_image)


if __name__ == "__main__":
    main()
