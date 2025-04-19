import sys
import os
import subprocess

# 定义路径
project_path = '/root/shared-nvme/zhenkui/code/WiXTAL/XRFV2'
dataset_root_path = '/root/shared-nvme/dataset/XRFV2'
causal_conv1d_path = '/root/shared-nvme/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
python_path = '/root/.conda/envs/mamba/bin/python'
# python_path = '/root/.conda/envs/t1/bin/python'
sys.path.append(project_path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")

import argparse
from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config, load_setting
from scripts.update_config import prepare_config
from global_config import get_basic_config

config = get_basic_config()

def init_configs():
    parser = argparse.ArgumentParser(description="WiVio study")
    parser.add_argument('--gpu', dest="gpu", required=False, type=int, default=0,
                        help="gpu")
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    day = get_day()
    args = init_configs()
    gpu = args.gpu
    # receivers_to_keep = {'imu': ['rh', 'rp', 'lh', 'lp', 'gl'], 'wifi': True, 'airpods': True, 'channel': (36, 270), 'modality': 'imu', 'tag': 'all'}
    receivers_to_keep = {'imu': None, 'wifi': True, 'airpods': True, 'channel': 270, 'modality': 'wifiimu', 'tag': '7'}
    
    # run_list = [
    #     ['github_code',   'TAD_muti_weight_grc', ('ActionFormer', 8, 250, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', '', (36, 270), 'wifi')]
    # ]
    run_list = [
        ['vim',   'TAD_muti_weight_grc', ('VisionMamba', 16, 100, {'layer': 8, 'i': 1}), ('WWADLDatasetMutiAll', '', 270, 'wifi')]
    ]

    for run in run_list:
        tag, model_arc_name, model_str, dataset_str, *_others = run
        dataset_name, dataset, channel, modality = dataset_str
        model_name, batch_size, epoch, model_config, *others = model_str
        # 调用提取的函数
        updated_config = prepare_config(model_arc_name, dataset_str, model_str, config, gpu, day, dataset_root_path, python_path, tag, receivers_to_keep=receivers_to_keep)

        test_gpu = gpu

        # TRAIN =============================================================================================
        run = Run_config(config, 'train')

        train_command = (
            f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} "
            f"{run.main_path} --is_train true --config_path {run.config_path}"
        )

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
