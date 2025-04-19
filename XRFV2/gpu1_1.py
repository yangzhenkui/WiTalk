import sys
import os
import torch
import subprocess

# project_path = '/home/lanbo/WWADL/WWADL_code'
# dataset_root_path = '/data/WWADL/dataset'

# 定义路径
project_path = '/root/shared-nvme/zhenkui/code/WiXTAL/XRFV2'
dataset_root_path = '/root/shared-nvme/dataset/XRFV2'
causal_conv1d_path = '/root/shared-nvme/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
python_path = '/root/.conda/envs/mamba/bin/python'
# python_path = '/root/.conda/envs/t1/bin/python'
sys.path.append(project_path)
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")

import argparse
from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config, load_setting
from scripts.update_config import prepare_config
from global_config import get_basic_config

config = get_basic_config()

receivers_to_keep_list = [
    {'imu': None, 'wifi': True, 'airpods': False, 'channel': 270, 'modality': 'wifi', 'tag': '7'},
    {'imu': None, 'wifi': True, 'airpods': False, 'channel': 270, 'modality': 'wifiimu', 'tag': '7'},
]

if __name__ == '__main__':
    day = get_day()
    gpu = 0
    run_list = {
        # 0: [
        #     ['device', receivers_to_keep_list[0], 'x', ('TemporalMaxer', 8, 100, {'layer': 8, 'i': 1}), ('WWADLDatasetSingle', '', (30, 270), 'wifi')]],
        0: [
            ['WiXTAL', receivers_to_keep_list[1], 'x', ('mamba', 16, 100, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', '', (30, 270), 'wifi')], 
            ['WiXTAL', receivers_to_keep_list[0], 'x', ('mamba', 16, 100, {'layer': 8, 'i': 1}), ('WWADLDatasetSingle', '', (30, 270), 'wifi')], 
            ]
    }

    for run in run_list[gpu]:
        tag, receivers_to_keep, model_arc_name, model_str, dataset_str, *_others = run
        dataset_name, dataset, channel, modality = dataset_str
        model_name, batch_size, epoch, model_config, *others = model_str
        tag = tag + receivers_to_keep['tag']

        if receivers_to_keep['modality'] == 'wifiimu':
            model_arc_name = 'TAD_muti_weight_grc'
        else:
            model_arc_name = 'TAD_single'


        # 调用提取的函数
        updated_config = prepare_config(model_arc_name, dataset_str, model_str, config, gpu, day, dataset_root_path, python_path, tag, receivers_to_keep=receivers_to_keep)

        test_gpu = gpu

        # TRAIN =============================================================================================
        run = Run_config(config, 'train')

        train_command = (
            f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} "
            f"{run.main_path} --is_train true --config_path {run.config_path}"
        )

        print(train_command)
        # exit(-1)

        # 执行训练命令并等待其完成
        train_process = subprocess.run(train_command, shell=True)
        

        # # 检查训练命令是否正常结束
        # if train_process.returncode == 0:  # 正常结束返回 0
        #     config = load_setting(os.path.join(config['path']['result_path'], 'setting.json'))
        #     config['endtime'] = get_time()
        #     write_setting(config)

        #     # TEST ==========================================================================================
        #     test_command = (
        #         f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
        #         f"{run.main_path} --config_path {run.config_path}"
        #     )

        #     # 启动测试命令
        #     subprocess.run(test_command, shell=True)
        # else:
        #     print("Training process failed. Test process will not start.")