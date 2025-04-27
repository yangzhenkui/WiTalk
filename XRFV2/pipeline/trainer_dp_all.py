import logging

import torch
import random
import os.path
import numpy as np
from tqdm import tqdm
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from strategy.loss.loss import MultiSegmentLoss
from dataset.wwadl import detection_collate

GLOBAL_SEED = 3407

logger = logging.getLogger(__name__)

# Worker 初始化函数
def worker_init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)

def register_hooks(model):
    for name, module in model.named_modules():
        # 仅对实际计算的模块注册钩子
        if not isinstance(module, (nn.Sequential, nn.ModuleList, nn.Identity)):
            module.register_forward_hook(forward_hook(name))

def _to_var(data: dict, device):
    for key, value in data.items():
        data[key] = value.to(device)  # Directly move tensor to device
    return data


def forward_hook(module_name):
    """
    钩子函数，用于检查输出是否合法。
    """
    def hook(module, input, output):
        # 检查输出是否是 Tensor 类型
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.error(f"NaN or Inf detected in module: {module_name}")
                raise RuntimeError(f"NaN or Inf detected in module: {module_name}")
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        logging.error(f"NaN or Inf detected in module: {module_name}, output[{i}]")
                        raise RuntimeError(f"NaN or Inf detected in module: {module_name}, output[{i}]")
        # 如果输出是 tuple 或 list，逐个检查其中的元素
        elif isinstance(output, dict):
            for key, out in output.items():
                if isinstance(out, torch.Tensor):
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        logging.error(f"NaN or Inf detected in module: {module_name}, key: {key}")
                        raise RuntimeError(f"NaN or Inf detected in module: {module_name}, key: {key}")
        else:
            logging.warning(f"Output of module: {module_name} is not a Tensor or tuple/list of Tensors. Skipping check.")
    return hook


class Trainer(object):
    def __init__(self,
                 config,
                 train_dataset,
                 model
                 ):
        super(Trainer, self).__init__()
        self.model = model
        self.train_dataset = train_dataset

        training_config = config['training']


        self.batch_size = training_config['train_batch_size']
        self.num_epoch = training_config['num_epoch']

        # loss setting -----------------------------------------------------------
        self.loss = MultiSegmentLoss(num_classes=config['model']['num_classes'], clip_length=config['dataset']['clip_length'])
        self.lw = config['loss']['lw']
        self.cw = config['loss']['cw']

        # learning config ---------------------------------------------------------
        self.opt_method = training_config['opt_method']
        self.lr_rate = training_config['lr_rate']
        self.lr_rate_adjust_epoch = training_config['lr_rate_adjust_epoch']
        self.lr_rate_adjust_factor = training_config['lr_rate_adjust_factor']
        self.weight_decay = training_config['weight_decay']


        self.check_point_path = config['path']['result_path']

        self.model_info = f'{config["model"]["backbone_name"]}_{config["model"]["model_set"]}'
        self.writer = SummaryWriter(os.path.join(self.check_point_path, f'tb_{self.model_info}'))

        # DDP setting -------------------------------------------------------------
        self.dist_url = 'env://'
        self.rank = 0
        self.world_size = 0
        self.gpu=0
        self.embeding_mode_name = config['embeding_mode_name']
        self.label_desc_type = config['label_desc_type']

        self.device = 'cuda'

    def _init_optimizer(self):

        params = self.model.parameters()

        if self.opt_method == 'adam':
            self.optimizer = torch.optim.Adam(params=params,
                                              lr=self.lr_rate,
                                              weight_decay=self.weight_decay)
        elif self.opt_method == 'adamw':
            self.optimizer = torch.optim.AdamW(params=params,
                                               lr=self.lr_rate,
                                               weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params=params,
                                             lr=self.lr_rate,
                                             weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         self.lr_rate_adjust_epoch,
                                                         self.lr_rate_adjust_factor)

    def _train_one_step(self, data, targets):
        data = _to_var(data, self.device)
        # data = data.to(self.device)  # 确保输入数据在正确的设备上
        targets = [t.to(self.device) for t in targets]
        self.optimizer.zero_grad()

        try:
            output_dict = self.model(data)
            assert not torch.isnan(output_dict['loc']).any(), "NaN detected in output_dict['loc']"
        except AssertionError as e:
            logging.info("Error occurred during training, saving data and model parameters for debugging...")

            # 创建保存路径
            error_save_path = os.path.join(self.check_point_path, "debug")
            os.makedirs(error_save_path, exist_ok=True)

            # 保存导致问题的输入数据
            data_save_path = os.path.join(error_save_path, "error_data.pt")
            torch.save(data, data_save_path)
            logging.info(f"Input data saved to: {data_save_path}")

            # 保存模型参数
            model_save_path = os.path.join(error_save_path, "error_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            logging.info(f"Model parameters saved to: {model_save_path}")

            # 再次抛出异常以中断训练
            raise e

        loc_p = output_dict['loc'].clamp(min=0)
        loss_l, loss_c = self.loss([loc_p, output_dict['conf'], output_dict["priors"][0]], targets)

        assert not torch.isnan(loss_l).any(), "NaN detected in loss_l"
        assert not torch.isnan(loss_c).any(), "NaN detected in loss_c"

        loss_l = loss_l * self.lw * 100
        loss_c = loss_c * self.cw

        loss = loss_l + loss_c

        # 反向传播
        loss.backward()

        # 优化器更新权重
        self.optimizer.step()

        # 无需分布式同步，直接返回损失
        return loss.item(), loss_l.item(), loss_c.item()

    def training(self):
        # 给不同的进程分配不同的、固定的随机数种子
        self.set_seed(3407)
        register_hooks(self.model)
        device = torch.device(self.device)

        # dataset loader -------------------------------------------------------------------------------
        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,  # DataParallel 不需要使用分布式采样
            pin_memory=True,
            num_workers=nw,  # 动态设置 workers
            collate_fn=detection_collate,  # 自定义 collate_fn
            worker_init_fn=worker_init_fn,  # 初始化每个 worker 的随机种子
            drop_last = True
        )

        # load model ------------------------------------------------------------------------------------
        # # 加载权重文件
        # checkpoint = torch.load("/root/shared-nvme/xrfv2_clip/logs/25_02-21/github_code/WWADLDatasetMutiAll__mamba_layer_8_i_1-7/mamba_mamba_layer_8_i_1-7-epoch-6.pt")

        # # 加载模型的state_dict
        # self.model.load_state_dict(checkpoint)

        # 转为DataParallel模型 ---------------------------------------------------------------------------
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        else:
            print("Using a single GPU for training")

        self.model = self.model.to(device=device)

        self._init_optimizer()

        for epoch in range(self.num_epoch):
            np.random.seed(epoch)  # 设置随机种子
            self.model.train()

            tbar = tqdm(train_loader, desc=f"{self.embeding_mode_name}_{self.label_desc_type}", ncols=100)

            iteration = 0
            loss_loc_val = 0
            loss_conf_val = 0
            cost_val = 0

            for clips, targets in tbar:
                iteration += 1
                loss, loss_l, loss_c = self._train_one_step(clips, targets)

                loss_loc_val += loss_l
                loss_conf_val += loss_c
                cost_val += loss

                tbar.set_description('Epoch: %d: ' % (epoch + 1))
                tbar.set_postfix(train_loss=loss)
                # 每次迭代清理显存
                torch.cuda.empty_cache()


            tbar.close()

            loss_loc_val /= (iteration + 1)
            loss_conf_val /= (iteration + 1)
            cost_val /= (iteration + 1)
            plog = 'Epoch-{} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}' \
                .format(epoch, cost_val, loss_loc_val, loss_conf_val)

            logging.info(plog)

            self.scheduler.step()

            # 保存当前模型
            if epoch > 50:
                # saver.save_model(self.model.state_dict(), f"{self.model_info}-epoch-{epoch}", cost_val)
                model_name = f"{self.model_info}-epoch-{epoch}"
                model_path = os.path.join(self.check_point_path, f"{model_name}.pt")
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved: {model_path} (Metric: {cost_val:.5f})")

            self.writer.add_scalar("Train Loss", cost_val, epoch)
            self.writer.add_scalar("loss_loc_val Loss", loss_loc_val, epoch)
            self.writer.add_scalar("loss_conf_val Loss", loss_conf_val, epoch)



    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic =True