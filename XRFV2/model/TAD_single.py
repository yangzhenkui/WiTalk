import torch
import torch.nn as nn
import logging
from model.models import register_model, make_backbone, make_backbone_config
from model.head import ClsLocHead
from model.TAD.embedding import Embedding
from model.model_config import TAD_single_Config
from model.embedding import TADEmbedding
from lib.VisionTransformer import ViT_wo_patch_embed
from functools import partial

logger = logging.getLogger(__name__)

import json
import torch

def load_json_to_tensor(filename: str,  target_key: str, return_type: str = "tensor",  model_key: str = "clip-vit-large-patch14"):

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
    

@register_model('TAD_single')
class TAD_single(nn.Module):
    def __init__(self, config: TAD_single_Config, label_desc_type="simple", model_key="clip-vit-large-patch14"):
        super(TAD_single, self).__init__()
        self.config = config
        if config.embed_type == 'Norm':
            self.embedding = Embedding(config.in_channels, stride=config.embedding_stride)
        else:
            self.embedding = TADEmbedding(config.in_channels, out_channels=512, layer=3, input_length=config.input_length)

        logger.info(f'load {config.embed_type} embedding')
        logger.info(f'load {config.backbone_name}')

        print("label_desc_type: ", label_desc_type)
        print("embeding_mode_name: ", model_key)

        self.embedding_mode_name = model_key
        self.embedding_mode_dim = model_dim_map[model_key]


        backbone_config = make_backbone_config(config.backbone_name, cfg=config.backbone_config)
        self.backbone = make_backbone(config.backbone_name, backbone_config)
        self.modality = config.modality
        self.head = ClsLocHead(num_classes=config.num_classes, head_layer=config.head_num)
        self.priors = []
        t = config.priors
        for i in range(config.head_num):
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        self.num_classes = config.num_classes
        self.text_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=self.embedding_mode_dim, depth=1,
                                                num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6))  # in: B*L*C
        
        self.label_desc_type = label_desc_type
        self.text_embeds = load_json_to_tensor("/root/shared-nvme/zhenkui/code/WiXTAL/xrf_v2.json", self.label_desc_type, model_key=self.embedding_mode_name)
        print("text_embeds: ", self.text_embeds.size())
        self.text_proj = nn.Sequential(nn.Linear(self.embedding_mode_dim, 2048))

    def fusion(self, text_features, wifi_features):
         # 元素级融合
        gated_text = 0.1 * text_features # (B, C, L)
        gated_wifi = 0.9 * wifi_features  # (B, C, L)

        combined_features = gated_text + gated_wifi

        return combined_features
    
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

        text_embeds = self.text_proj(text_embeds)
        
        # Projection处理
        # 调整到目标维度 (B, 512, 2048)
        current_branches = text_embeds.size(1)
        repeat_factor = (270 + current_branches - 1) // current_branches
        text_embeds = text_embeds.repeat(1, repeat_factor, 1)
        text_embeds = text_embeds[:, :270, :]

        # 沿着第一维（batch 维度）取均值
        text_embeds = text_embeds.mean(dim=0)
        return text_embeds
    
    def forward(self, input):
        x = input[self.modality]
        B, C, L = x.size()

        # to use WiFi single modality, simply comment out the text-related code.
        x_text = self.cal_text_features_2d()
        # print(x.shape, x_text.shape)

        x = self.fusion(x_text, x)

        x = self.embedding(x)
        feats = self.backbone(x)

        out_offsets, out_cls_logits = self.head(feats)
        priors = torch.cat(self.priors, 0).to(x.device).unsqueeze(0)
        loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
        conf = torch.cat([o.view(B, -1, self.num_classes) for o in out_cls_logits], 1)

        return {
            'loc': loc,
            'conf': conf,
            'priors': priors # trainer ddp需要弄成priors[0]
        }
