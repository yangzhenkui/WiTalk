import os
import sys
import json
import subprocess
import multiprocessing

# ======== 根据实际项目环境，修改这些路径 ========
project_path = '/root/shared-nvme/zhenkui/code/WiXTAL/XRFV2'
dataset_root_path = '/root/shared-nvme/WWADL'
causal_conv1d_path = '/root/shared-nvme/video-mamba-suite/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
python_path = '/root/.conda/envs/mamba/bin/python'

# 将项目路径加到 sys.path
sys.path.append(project_path)

# 设置 PYTHONPATH 环境变量
os.environ["PYTHONPATH"] = (
    f"{project_path}:{causal_conv1d_path}:{mamba_path}:"
    + os.environ.get("PYTHONPATH", "")
)

from utils.setting import (
    get_day, get_time, write_setting, get_result_path,
    get_log_path, Run_config
)

def load_setting(url: str) -> dict:
    """从指定 JSON 文件中加载配置。"""
    with open(url, 'r') as f:
        data = json.load(f)
    return data

# 需要在不同 GPU 上顺序执行的任务列表
# key: GPU 编号, value: 这个 GPU 上要顺序执行的多个模型路径列表
run_tasks = {
    0: [
        "/root/shared-nvme/zhenkui/code/WiXTAL/XRFV2/logs/25_03-30/qwen_text-embedding-v3_simple/WWADLDatasetSingle__mamba_layer_8"
    ]
}

def run_tasks_on_one_gpu(gpu_id: int, model_paths: list):
    """
    在单块 GPU (gpu_id) 上，顺序执行 model_paths 中的多个任务。
    """
    for test_model_path in model_paths:
        # 1) 读取 setting.json
        json_path = os.path.join(test_model_path, 'setting.json')
        config = load_setting(json_path)

        # 2) 修改 config 中的字段
        config['path']['dataset_root_path'] = '/root/shared-nvme/WWADL'
        config['path']['dataset_path'] = '/root/shared-nvme/dataset/XRFV2'
        config['path']['basic_path']['python_path'] = '/root/.conda/envs/mamba/bin/python'

        # 3) 用修改好的 config 初始化 Run_config
        run_cfg = Run_config(config, 'train')

        # 如果你需要将修改过的设置写回 setting.json，就调用 write_setting
        write_setting(config)

        # 4) 打印信息
        print(f"\n[GPU {gpu_id}] 即将在 {test_model_path} 执行任务...")
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} {python_path} "
            f"{run_cfg.main_path} --config_path {run_cfg.config_path}"
        )
        print("执行命令:", cmd)

        # 5) 用阻塞方式运行，保证在同一 GPU 上是顺序执行
        subprocess.run(cmd, shell=True, check=True)


def main():
    # 存储各 GPU 的进程
    processes = []

    # 为每块 GPU 启动一个进程
    for gpu, model_paths in run_tasks.items():
        p = multiprocessing.Process(
            target=run_tasks_on_one_gpu,
            args=(gpu, model_paths)
        )
        p.start()
        processes.append(p)

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("所有 GPU 上的任务都执行完毕！")


if __name__ == "__main__":
    main()
