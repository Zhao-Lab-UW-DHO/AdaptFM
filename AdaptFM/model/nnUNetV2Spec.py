from pathlib import Path

from AdaptFM.model.model_spec import ModelSpec
import os
import subprocess
import json

class NNUNetV2ModelSpec(ModelSpec):
    
    def __init__(self,conda_env):
        super().__init__()
        self.conda_env = conda_env
        self.name = 'nnUNetV2'


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


    def prepare_dataset(self, dataset_manager, output_dir,params):
       
        dataset_manager.export_for_framework(
            framework="nnunet",
            out_folder=output_dir,
            params=params
        )
        return output_dir
    
    def preprocessing_command(self, params):
        set_id = params["Set ID"]

        return [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(set_id), 
            "-pl", "nnUNetPlannerResEncL", # use default plans
        ]
    

    def run_preprocessing(self, dataset_dir, params, output_dir):
        
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu = params['gpu']

        cmd = self._wrap_with_conda(
            self.preprocessing_command( params)
        )

        env = os.environ.copy()
        env['nnUNet_raw'] = str(Path(dataset_dir) / 'nnUNet_raw')
        env['nnUNet_preprocessed'] = str(Path(dataset_dir) / 'nnUNet_preprocessed')
        env['nnUNet_results'] = str(Path(dataset_dir) / 'nnUNet_results')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        process= subprocess.Popen(
            cmd,
            stdout=(output_dir / "stdout.log").open(mode='w'),
            stderr=(output_dir / "stderr.log").open(mode='w'),
            start_new_session=True,
            env=env
        )

        return process


    def training_command(self, params):
        return [
            'nnUNetv2_train',
            params['Set ID'],
            params['config'],
            params['fold'],
            '-p', 'nnUNetResEncUNetLPlans', 
            '-tr', 'nnUNetTrainer',
            '--npz'
        ]


     # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        gpu = params['gpu']
        env = os.environ.copy()
        env['nnUNet_raw'] = str(Path(dataset_info) / 'nnUNet_raw')
        env['nnUNet_preprocessed'] = str(Path(dataset_info) / 'nnUNet_preprocessed')
        env['nnUNet_results'] = str(Path(dataset_info) / 'nnUNet_results')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        with (Path(run_dir) / 'params.json').open('w') as f:
            json.dump(params, f, indent=4)

        training_commnad = self.training_command(params)

        cmd = self._wrap_with_conda(
            training_commnad
        )
        env['TORCHDYNAMO_DISABLE'] = '1'

        subprocess.Popen(
            cmd,
            stdout=(run_dir / "stdout.log").open(mode='w'),
            stderr=(run_dir / "stderr.log").open(mode='w'),
            start_new_session=True,
            env=env
        )

        
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params['gpu']
        env = os.environ.copy()

        env['nnUNet_raw'] = str(Path(dataset_dir) / 'nnUNet_raw')
        env['nnUNet_preprocessed'] = str(Path(dataset_dir) / 'nnUNet_preprocessed')
        env['nnUNet_results'] = str(Path(dataset_dir) / 'nnUNet_results')

        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)      

        inference_cmd = self.inference_command(params = params,
                                               output_dir=output_dir,
                                               checkpoint = checkpoint,
                                               env=env) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=(output_dir / "stdout.log").open(mode='w'),
            stderr=(output_dir / "stderr.log").open(mode='w'),
            start_new_session=True,
            env=env
        )


    def inference_command(self, params, output_dir,checkpoint, env):
        
        imagesTs = str(Path(env['nnUNet_raw']) / f'Dataset{params['Set ID']}_{params['Set Name']}' / 'imagesTs')
        ckpt_name = Path(checkpoint).name

        return [
            'nnUNetv2_predict',
            '-i',imagesTs,
            '-o',output_dir,
            '-d',params['Set ID'],
            '-c',params['config'],
            '-f',params['fold'],
            '-p','nnUNetResEncUNetLPlans', # use as default plans
            '-tr', 'nnUNetTrainerCELoss', # default trainer
            '-chk',ckpt_name # use the best checkpoint by default 
        ]
