import sys
import os
import torch
import subprocess


# 定义路径
project_path = '/root/shared-nvme/zhenkui/code/WiXTAL/XRFV2'
dataset_root_path = '/root/shared-nvme/dataset/XRFV2'
causal_conv1d_path = '/root/shared-nvme/video-mamba-suite/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
python_path = '/root/.conda/envs/mamba/bin/python'
sys.path.append(project_path)
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")


from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config, load_setting
from global_config import get_basic_config

config = get_basic_config()


if __name__ == '__main__':

    day = get_day()

    model = 'TAD_single'
    gpu = 0

    model_str_list = [
        ('mamba', 16, 80, {'layer': 8, 'i':2}),
        # ('Transformer', 16, 80, {'layer': 8}),
        # ('Transformer', 16, 80, {'layer': 8, 'embed_type': 'Norm'}),
        # ('wifiTAD', 16, 80, {'layer': 8}),
    ]

    dataset_str_list = [
        ('WWADLDatasetSingle', '', 270, 'wifi'),
    ]

    for dataset_str in dataset_str_list:
        dataset_name, dataset, channel, modality = dataset_str
        for model_str in model_str_list:
            model_name, batch_size, epoch, model_config, *others = model_str
            model_set = model_name
            for k, v in model_config.items():
                model_set += f'_{k}_{v}'
            config['datetime'] = get_time()
            config["training"]["DDP"]["enable"] = True
            config["training"]["DDP"]["devices"] = [gpu]
            config["model"]['num_classes'] = 30
            config["dataset"]['num_classes'] = 30
            config["model"]['name'] = model
            config["model"]["backbone_config"] = model_config
            config["model"]["backbone_name"] = model_name

            if 'embed_type' in model_config:
                config['model']['embed_type'] = model_config['embed_type']

            if isinstance(channel, tuple):
                config["model"]['imu_in_channels'] = channel[0]
                config["model"]['wifi_in_channels'] = channel[1]
            else:
                config["model"]["in_channels"] = channel
            config["model"]["model_set"] = model_set
            config["model"]["modality"] = modality
            config["training"]["lr_rate"] = 4e-05


            test_gpu = gpu
            
            label_desc_type = config['label_desc_type']
            embeding_mode_name = config['embeding_mode_name']
            # TAG ===============================================================================================
            tag = f'{embeding_mode_name}_{label_desc_type}'

            config['path']['dataset_path'] = os.path.join(dataset_root_path, dataset)
            config['path']['log_path']      = get_log_path(config, day, f'{dataset_name}_{dataset}', model_set, tag)
            config['path']['result_path']   = get_result_path(config, day, f'{dataset_name}_{dataset}', model_set, tag)

            config['dataset']['dataset_name'] = dataset_name
            config['dataset']['clip_length'] = 1500

            config["training"]['num_epoch'] = epoch
            config["training"]['train_batch_size'] = batch_size

            write_setting(config)

            # TRAIN =============================================================================================
            run = Run_config(config, 'train')

            # os.system(
            #     f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} -m torch.distributed.launch --nproc_per_node {run.nproc_per_node} "
            #     f"--master_port='29501' --use_env "
            #     f"{run.main_path} --is_train true --config_path {run.config_path}"
            # )
            # TRAIN =============================================================================================
            train_command = (
                f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} "
                f"{run.main_path} --is_train true --config_path {run.config_path}"
            )

            # print(train_command)
            # exit(-1)

            # 执行训练命令并等待其完成
            train_process = subprocess.run(train_command, shell=True)

            # 检查训练命令是否正常结束
            if train_process.returncode == 0:  # 正常结束返回 0
                config = load_setting(os.path.join(config['path']['result_path'], 'setting.json'))
                config['endtime'] = get_time()
                write_setting(config)

                # TEST ==========================================================================================
                test_command = (
                    f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
                    f"{run.main_path} --config_path {run.config_path}"
                )

                # 启动测试命令
                subprocess.run(test_command, shell=True)
            else:
                print("Training process failed. Test process will not start.")