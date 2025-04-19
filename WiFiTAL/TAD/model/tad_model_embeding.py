import torch
import torch.nn as nn
import numpy as np
from TAD.model.embedding import Embedding
from TAD.model.module import ScaleExp
from TAD.model.backbone import TSSE, LSREF
from TAD.model.head import PredictionHead
from TAD.config import config
from lib.VisionTransformer import ViT_wo_patch_embed
from functools import partial
num_classes = config['dataset']['num_classes']
layer_num = 4
skip_ds_layer = 4
priors = 128


class Pyramid_Detection(nn.Module):
    def __init__(self):
        super(Pyramid_Detection, self).__init__()
        self.layer_skip = skip_ds_layer
        self.skip_tsse = nn.ModuleList()
        
        
        self.layer_num = layer_num
        self.PyTSSE = nn.ModuleList()
        self.PyLSRE = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        
        for i in range(self.layer_skip):
            self.skip_tsse.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=2048//(2**i)))
        
        
        for i in range(layer_num):
            self.PyTSSE.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=priors//(2**i)))
            self.PyLSRE.append(LSREF(len=priors//(2**i),r=(2048//priors)*(2**i)))
            
        self.PredictionHead = PredictionHead()

        self.priors = []
        t = priors
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        
    def forward(self, embedd):
        
        deep_feat = embedd
        global_feat = embedd.detach()
        for i in range(len(self.skip_tsse)):
            deep_feat = self.skip_tsse[i](deep_feat)
        
        
        batch_num = deep_feat.size(0)
        out_feats = []
        locs = []
        confs = []
        for i in range(self.layer_num):
            deep_feat = self.PyTSSE[i](deep_feat)
            out = self.PyLSRE[i](deep_feat, global_feat)
            out_feats.append(out)
        
        for i, feat in enumerate(out_feats):
            loc_logits, conf_logits = self.PredictionHead(feat)
            locs.append(
                self.loc_heads[i](loc_logits)
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )
            confs.append(
                conf_logits.view(batch_num, num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, priors

import json
import torch

def load_json_to_tensor(filename: str,  target_key: str, return_type: str = "tensor",  model_key: str = "llama"):

    with open(filename, "r") as f:
        data = json.load(f)
    
    # 提取指定模型的数据
    res = data[model_key]
    
    if target_key not in res:
        raise KeyError(f"Key '{target_key}' not found in {model_key} data")
    
    if return_type.lower() == "tensor":
        tensor_value = torch.tensor(res[target_key]).to("cuda:0")
        return tensor_value
    else:
        # 直接返回指定key的原始值
        return res[target_key]
    
model_dim_map = {
    "qwen_text-embedding-v3": 768,
    "llama": 4096,
    'clip-vit-large-patch14': 512,
    "t5-small": 512,
    "xlm-roberta-base": 768
}

class wifitad_text(nn.Module):
    def __init__(self, embed_type="simple", model_key="t5-small"):


        super(wifitad_text, self).__init__()
        self.embedding = Embedding(30)
        self.pyramid_detection = Pyramid_Detection()
        self.reset_params()
        self.embed_dim = model_dim_map[model_key]

        self.text_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=self.embed_dim, depth=1,
                                                num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6))  # in: B*L*C
        

        self.embed_type = embed_type
        self.text_embeds = load_json_to_tensor("/root/shared-nvme/zhenkui/code/WiXTAL/wifiTAD.json", self.embed_type, model_key=model_key)
        self.text_proj = nn.Sequential(nn.Linear(self.embed_dim, 4096))
        
    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)
    
    def cal_text_features_2d(self, device="cuda:0"):
        """
        从JSON文件加载text embedding并处理
        
        Args:
            json_file (str): JSON文件路径
            target_key (str): 要加载的特定key
            device (str): 设备，默认"cuda:0"
        """
        # 直接从JSON加载embedding
        text_embeds = self.text_embeds.to(device)
    
        # 如果需要调整维度
        if len(text_embeds.shape) == 2:
            text_embeds = text_embeds.unsqueeze(1)  # 添加branches维度

        text_embeds_att= self.text_self_attention(text_embeds)
        # 拼接处理
        text_embeds = torch.cat([text_embeds, text_embeds_att.unsqueeze(1)], dim=1)
        
        if text_embeds.shape[2] != 4096:
            text_embeds = self.text_proj(text_embeds)
        
        # Projection处理
        # 调整到目标维度 (B, 512, 2048)
        current_branches = text_embeds.size(1)
        repeat_factor = (512 + current_branches - 1) // current_branches
        text_embeds = text_embeds.repeat(1, repeat_factor, 1)
        text_embeds = text_embeds[:, :512, :]
        text_embeds = text_embeds.mean(dim=0)
        
        return text_embeds

    def fusion(self, x_text, x_wifi):
        # 元素级融合
        gated_text = 0.1 * x_text  # (B, C, L)
        gated_wifi = 0.9 * x_wifi  # (B, C, L)

        combined_features = gated_text + gated_wifi

        return combined_features
    
    def forward(self, x):
        x_wifi = self.embedding(x)
        x_text = self.cal_text_features_2d()
        x = self.fusion(x_text, x_wifi)
        loc, conf, priors = self.pyramid_detection(x)
        return {
            'loc': loc,
            'conf': conf,
            'priors': priors
        }
