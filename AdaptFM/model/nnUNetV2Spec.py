from AdaptFM.model.model_spec import ModelSpec
import os
import subprocess
import json
from pathlib import Path
import shutil

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
       
        prepared_dataset_folder = dataset_manager.export_for_framework(
            framework="nnunet",
            out_folder=output_dir,
            params=params
        )
        return prepared_dataset_folder
    
    def preprocessing_command(self, params):
        set_id = params["Set ID"]

        return [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(set_id)]
    

    def run_preprocessing(self, dataset_dir, params, output_dir):
        
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu = params['gpu']

        env = os.environ.copy()
        env['nnUNet_raw'] = os.path.join(dataset_dir,'nnUNet_raw')
        env['nnUNet_preprocessed'] = os.path.join(dataset_dir,'nnUNet_preprocessed')
        env['nnUNet_results'] =os.path.join(dataset_dir,'nnUNet_results')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)


        cmd = self._wrap_with_conda(
                    self.preprocessing_command(params)
                )

        process= subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
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
            '-p', 'nnUNetPlans', 
            '-tr', 'nnUNetTrainer',
            '--npz'
        ]


     # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        gpu = params['gpu']
        env = os.environ.copy()
        env['nnUNet_raw'] = os.path.join(dataset_info,'nnUNet_raw')
        env['nnUNet_preprocessed'] = os.path.join(dataset_info,'nnUNet_preprocessed')
        env['nnUNet_results'] =os.path.join(dataset_info,'nnUNet_results')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        training_commnad = self.training_command(params)

        cmd = self._wrap_with_conda(
            training_commnad
        )
        env['TORCHDYNAMO_DISABLE'] = '1'

        subprocess.Popen(
            cmd,
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=open(run_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

        
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params['gpu']
        env = os.environ.copy()

        env['nnUNet_raw'] = os.path.join(dataset_dir,'nnUNet_raw')
        env['nnUNet_preprocessed'] = os.path.join(dataset_dir,'nnUNet_preprocessed')
        env['nnUNet_results'] =os.path.join(dataset_dir,'nnUNet_results')

        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)      

        inference_cmd = self.inference_command(params = params,
                                               output_dir=output_dir,
                                               checkpoint = checkpoint,
                                               env=env) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )


    def inference_command(self, params, output_dir,checkpoint, env):
        
        imagesTs = os.path.join(env['nnUNet_raw'],f'Dataset{params['Set ID']}_{params['Set Name']}','imagesTs')
        ckpt_name = os.path.basename(checkpoint)

        return [
            'nnUNetv2_predict',
            '-i',imagesTs,
            '-o',output_dir,
            '-d',params['Set ID'],
            '-c',params['config'],
            '-f',params['fold'],
            '-p','nnUNetPlans', # use as default plans
            '-tr', 'nnUNetTrainer', # default trainer
            '-chk',ckpt_name # use the best checkpoint by default 
        ]
    

class MerlinNNUNetV2ModelSpec(NNUNetV2ModelSpec):
    def __init__(self,conda_env,transform_path):
        super().__init__(conda_env)
        self.name = "Merlin nnUNet"
        self.transform_path = transform_path


    def ensure_checkpoint_matches_dataset(self,checkpoint: str, output_dir: Path, params: dict):
        setID = params["Set ID"]
        setName = params["Set Name"]

        formatted_set_id = f"{int(setID):03d}"
        setName_and_ID = f"Dataset{formatted_set_id}_{setName}"

        # 1. Check if checkpoint contains the dataset tag
        if setName_and_ID in str(checkpoint):
            return  # nothing to do

        # 2. Create nnUNet folder structure
        raw_folder = output_dir / "nnUNet_raw" / setName_and_ID
        preprocessed_folder = output_dir / "nnUNet_preprocessed" / setName_and_ID
        results_folder = output_dir / "nnUNet_results" / setName_and_ID

        raw_folder.mkdir(parents=True, exist_ok=True)
        preprocessed_folder.mkdir(parents=True, exist_ok=True)
        results_folder.mkdir(parents=True, exist_ok=True)

        # 3. Locate the nnUNetTrainerMerlin folder inside the checkpoint path
        checkpoint_path = Path(checkpoint).resolve()

        # Search upward for the folder named nnUNetTrainerMerlin__nnUNetPlans__3d_fullres
        trainer_folder = None
        for parent in checkpoint_path.parents:
            candidate = parent / "nnUNetTrainerMerlin__nnUNetPlans__3d_fullres"
            if candidate.exists() and candidate.is_dir():
                trainer_folder = candidate
                break

        if trainer_folder is None:
            raise RuntimeError(
                "Could not find nnUNetTrainerMerlin__nnUNetPlans__3d_fullres folder "
                "in checkpoint path parents."
            )

        # 4. Move trainer folder into nnUNet_results/DatasetXXX_Name
        destination = results_folder / trainer_folder.name
        if destination.exists():
            # do not overwrite
            return

        shutil.copytree(str(trainer_folder), str(destination))


        
    def run_preprocessing(self, dataset_dir, params, output_dir):
            
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu = params['gpu']

        env = os.environ.copy()
        env['nnUNet_raw'] = os.path.join(dataset_dir,'nnUNet_raw')
        env['nnUNet_preprocessed'] = os.path.join(dataset_dir,'nnUNet_preprocessed')
        env['nnUNet_results'] =os.path.join(dataset_dir,'nnUNet_results')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)


        setID = params['Set ID']
        setName = params["Set Name"] 

        folder2transform = dataset_dir / 'nnUNet_raw' / f"Dataset{setID:03}_{setName}" / "imagesTr"
 
        transform_cmd = self._wrap_with_conda(
            self.transform_command(folder2transform,params)
        )

        transform_proc= subprocess.Popen(
            transform_cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )
        
                    # Block until preprocessing is done
        return_code = transform_proc.wait()
        
        if return_code != 0:
            raise RuntimeError(f"Transforms failed with return code {return_code}")



        cmd = self._wrap_with_conda(
                    self.preprocessing_command(params)
                )

        process= subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
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
            '-p', 'nnUNetPlans', 
            '-tr', 'nnUNetTrainerMerlin', #training command is identical just uses a different trainer
            '--npz'
        ]


    def transform_command(self,folder2transform,params):
        
        return[
            "python",
            "-m", f"{self.transform_path}",
            "--params", json.dumps(params),
            "--folder2transform", str(folder2transform)

        ]
    

    def inference_command(self, params,imagesTs, output_dir,checkpoint):
        
        ckpt_name = os.path.basename(checkpoint)

        return [
            'nnUNetv2_predict',
            '-i',imagesTs,
            '-o',output_dir,
            '-d',params['Set ID'],
            '-c',params['config'],
            '-f',params['fold'],
            '-p','nnUNetPlans', # use as default plans
            '-tr', 'nnUNetTrainerMerlin', # default trainer
            '-chk',ckpt_name # use the best checkpoint by default 
        ]
    
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params['gpu']
        env = os.environ.copy()

        env['nnUNet_raw'] = os.path.join(output_dir,'nnUNet_raw')
        env['nnUNet_preprocessed'] = os.path.join(output_dir,'nnUNet_preprocessed')
        env['nnUNet_results'] =os.path.join(output_dir,'nnUNet_results')

        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)     


        #ensure that checkpoint is in the set ID,setName results folder. if not, copy it there
        self.ensure_checkpoint_matches_dataset(checkpoint, output_dir, params)

        folder2transform = dataset_dir
 
        transform_cmd = self._wrap_with_conda(
            self.transform_command(folder2transform,params)
        )

        transform_proc= subprocess.Popen(
            transform_cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )
        

        # Block until transforms are done
        return_code = transform_proc.wait()
        
        if return_code != 0:
            raise RuntimeError(f"Transforms failed with return code {return_code}")
        
        original_name = folder2transform.name
        transformed_name = f"{original_name}_transformed"

        transformed_folder = folder2transform.parent /transformed_name        
        
        inference_cmd = self.inference_command(params = params,
                                               imagesTs=str(transformed_folder),
                                               output_dir=output_dir,
                                               checkpoint = checkpoint) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

