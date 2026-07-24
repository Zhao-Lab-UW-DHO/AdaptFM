import argparse
import rescale4dl

def run_postprocessing(input_dir: str):


    rescale4dl.analyse(input_dir,
            is_3d=True,
            run_per_object_stats = False, # True for Instance Segmentation, False for Semantic or Binary Segmentation
            sampling_dir_list = None,
            save_images=True)
    

    
if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")


    args = parser.parse_args()

    run_postprocessing(str(args.input_dir))