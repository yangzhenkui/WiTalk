import torch
import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, XLMRobertaModel
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from functools import partial
import sys, os
sys.path.append('.')
from lib.VisionTransformer import ViT_wo_patch_embed

class TextEmbeding(nn.Module):
    def __init__(self, proj_head_dim=64, if_use_hm_proj=False, if_use_text_proj=False , device="cuda:0", 
                                        default_model="t5-base"):
        super().__init__()
        self.device = device

        
        self.text_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=2048, depth=1,
                                                          num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                          norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
        
        self.if_use_hm_proj = if_use_hm_proj
        self.if_use_text_proj = if_use_text_proj
        if if_use_hm_proj:
            self.hm_proj = nn.Sequential(nn.Linear(512, proj_head_dim)).to(device)
        if if_use_text_proj:
            self.text_proj = nn.Sequential(nn.Linear(4096, proj_head_dim)).to(device)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.final_proj = nn.Linear(512, 4096).to(device)

        # 模型配置
        self.model_configs = {
            "all-MiniLM-L6-v2": {
                "path": "/root/shared-nvme/zhenkui/code/WiXTAL/WiFiTAD/all-MiniLM-L6-v2",
                "encoder": None,
                "processor": None,
                "max_length": 256
            },
            "t5-base": {
                "path": "/root/shared-nvme/zhenkui/code/WiXTAL/WiFiTAD/t5-base",
                "encoder": None,
                "processor": None,
                "max_length": 512
            },
            "t5-small": {
                "path": "/root/shared-nvme/zhenkui/code/WiXTAL/WiFiTAD/t5-small",
                "encoder": None,
                "processor": None,
                "max_length": 512
            },
            "xlm-roberta-base": {
                "path": "/root/shared-nvme/zhenkui/code/WiXTAL/WiFiTAD/xlm-roberta-base",
                "encoder": None,
                "processor": None,
                "max_length": 512
            },
            "word2vec": {
                "path": "/root/shared-nvme/zhenkui/code/WiXTAL/WiFiTAD/word2vec",
                "encoder": None,
                "processor": None,
                "max_length": None
            }
        }

        # 按需加载逻辑
        self._ensure_model_loaded(default_model)  # 初始化时加载默认模型

    def _ensure_model_loaded(self, model_type):
        """按需加载模型，确保只加载一次"""
        config = self.model_configs[model_type]
        if config["encoder"] is None:
            print(f"Loading model {model_type} from {config['path']} ...")
            if model_type == "all-MiniLM-L6-v2":
                # 确保路径是本地目录
                local_path = config["path"]
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"本地路径 {local_path} 不存在，请检查模型文件是否正确下载并放置。")
                config["encoder"] = SentenceTransformer(local_path, local_files_only=True).to(self.device).eval()
                config["processor"] = None
            elif model_type in ["t5-base", "t5-small"]:
                config["encoder"] = T5EncoderModel.from_pretrained(config["path"]).to(self.device).eval()
                config["processor"] = T5Tokenizer.from_pretrained(config["path"])
            elif model_type == "xlm-roberta-base":
                config["encoder"] = XLMRobertaModel.from_pretrained(config["path"]).to(self.device).eval()
                config["processor"] = AutoTokenizer.from_pretrained(config["path"])
            elif model_type == "word2vec":
                config["encoder"] = Word2Vec.load(config["path"]).wv
                config["processor"] = lambda x: x.split()
            for param in config["encoder"].parameters() if hasattr(config["encoder"], "parameters") else []:
                param.requires_grad = False

    def cal_text_features_2d(self, text_list_2d, model_type="t5-base", device="cuda:0", out_dim=270):
        """计算文本嵌入，支持多种模型"""
        self._ensure_model_loaded(model_type)  # 确保模型已加载
        
        config = self.model_configs[model_type]
        encoder = config["encoder"]
        processor = config["processor"]
        max_length = config["max_length"]

        length = len(text_list_2d)
        text_branches = len(text_list_2d[0])
        text_list = [item for sublist in text_list_2d for item in sublist]

        with torch.no_grad():
            if model_type == "word2vec":
                text_embeds = []
                for text in text_list:
                    words = processor(text)
                    word_vecs = [encoder[word] for word in words if word in encoder]
                    text_embeds.append(np.mean(word_vecs, axis=0) if word_vecs else np.zeros(300))
                text_embeds = torch.tensor(text_embeds, dtype=torch.float32).to(device)
            else:
                inputs = processor(text_list, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device) if processor else text_list
                if model_type == "all-MiniLM-L6-v2":
                    text_embeds = encoder.encode(text_list, convert_to_tensor=True).to(device)
                elif model_type in ["t5-base", "t5-small"]:
                    text_embeds = encoder(**inputs).last_hidden_state.mean(dim=1)
                else:  # xlm-roberta-base
                    text_embeds = encoder(**inputs).last_hidden_state.mean(dim=1)

        text_embeds = text_embeds.reshape(length, text_branches, -1).to(device)

        proj_to_attn = nn.Linear(text_embeds.size(-1), 2048).to(device)
        text_embeds = proj_to_attn(text_embeds)
        text_embeds_att, _ = self.text_self_attention(text_embeds)

        text_embeds = torch.cat([text_embeds, text_embeds_att.unsqueeze(1)], dim=1)

        text_embeds = text_embeds.repeat(1, (out_dim + text_embeds.size(1) - 1) // text_embeds.size(1), 1)  # 扩展到接近512
        
        text_embeds = text_embeds[:, :out_dim, :]  # 裁剪到 512

        return text_embeds  # 输出 B*512*4096

