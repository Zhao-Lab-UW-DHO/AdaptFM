from AdaptFM.model.model_spec import ModelSpec
import os
import subprocess
import json

class NNUNetV2ModelSpec(ModelSpec):
    name = "nnUNetv2"
    conda_env = "/Organoids/"
    python_env = '/home/wisc/hbakhtiar/Organoids'


    def default_params(self):
        return {
            "config": "3d_fullres",
            "fold": "all",
        }

    def tunable_params(self, **kwargs):
        # kwargs intentionally ignored
        return {


            "Set Name":{
                'type': str,
                'default':''

            },

            "Set ID":{
                "type": str,
                'default': '1'

            },

            "config": {
                "type": str,
                "choices": ["2d", "3d_fullres"],
                "default": "3d_fullres",
            },
            "fold": {
                "type": str,
                "default": "all",
            },
        }


    def prepare_dataset(self, dataset_manager, output_dir):
        dataset_manager.export_for_framework(
            framework="nnunet",
            out_folder=output_dir
        )
        return output_dir
    
    def preprocessing_command(self, dataset_dir, params):
        set_id = params["Set ID"]

        return [
            "/home/wisc/hbakhtiar/Organoids/bin/nnUNetv2_plan_and_preprocess",
            "-d", str(set_id),
            "-pl", "nnUNetPlannerResEncL",
        ]

    def training_command(self, dataset_dir, params, run_dir):
        return [
            '/home/wisc/hbakhtiar/Organoids/bin/nnUNetv2_train',
            params['Set ID'],
            params['config'],
            params['fold'],
            '-p', 'nnUNetResEncUNetLPlans',
            '-tr', 'nnUNetTrainerCELoss',
            '--npz'
        ]

    def inference_command(self, params, images_dir, output_dir,checkpoint):
    
         return [
            '/home/wisc/hbakhtiar/Organoids/bin/nnUNetv2_train',
            '-i',images_dir,
            '-o',output_dir,
            '-d',params['Set ID'],
            '-c',params["config"],
            '-f',params['fold'],
            '-p','nnUNetResEncUNetLPlans',
            '-tr', 'nnUNetTrainerCELoss',
            '-chk',checkpoint.basename
        ]
    
    def run_preprocessing(self, dataset_dir, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._wrap_with_python_env(
            self.preprocessing_command(dataset_dir, params)
        )

        env = os.environ.copy()
        env['nnUNet_raw'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_raw'
        env['nnUNet_preprocessed'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_preprocessed'
        env['nnUNet_results'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_results'
        gpu = params.pop("gpu", None)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        subprocess.run(
            cmd,
            check=True,
            env=env,
        )


     # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        cmd = self._wrap_with_python_env(
            self.training_command(dataset_info, params, run_dir)
        )

        env = os.environ.copy()
        env['nnUNet_raw'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_raw'
        env['nnUNet_preprocessed'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_preprocessed'
        env['nnUNet_results'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_results'
        gpu = params.pop("gpu", None)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        subprocess.Popen(
            cmd,
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=open(run_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

        
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params.pop("gpu", None)
        env = os.environ.copy()

        env['nnUNet_raw'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_raw'
        env['nnUNet_preprocessed'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_preprocessed'
        env['nnUNet_results'] = '/mnt/local/data3/Organoids/Data/nnUNet_testing_results/dataset/nnUNet_results'
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)      

        inference_cmd = self.inference_command(dataset_dir=dataset_dir,
                                               checkpoint=checkpoint,
                                               output_dir=output_dir) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )
