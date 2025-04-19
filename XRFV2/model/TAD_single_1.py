import torch
import torch.nn as nn
import logging
from model.models import register_model, make_backbone, make_backbone_config
from model.head import ClsLocHead
from model.TAD.embedding import Embedding
from model.model_config import TAD_single_Config
from model.embedding import TADEmbedding
from model.TextEmbeding import TextEmbeding

logger = logging.getLogger(__name__)

labels = ["The system can recognize actions such as stretching, pouring water, writing, cutting fruit, eating fruit, and taking medicine.",
"It also detects actions like drinking water, sitting down, turning on/off the eye protection lamp, and opening/closing curtains.",
"The system can identify activities such as opening/closing windows, typing, opening envelopes, throwing garbage, and picking fruit.",
"Other actions include picking up items, answering the phone, using a mouse, wiping the table, and writing on the blackboard.",
"It also recognizes actions like washing hands, using a phone, reading, watering plants, and walking to different locations (e.g., bed, chair, cabinet, window, blackboard).",
"Finally, the system can detect movements like getting out of bed, standing up, lying down, standing still, and lying still."]


labels = [
    "The system uses Wi-Fi signals to recognize actions such as stretching, pouring water, writing, cutting fruit, eating fruit, and taking medicine.",
    "It can also detect movements like drinking water, sitting down, turning on/off the eye protection lamp, opening/closing curtains, and opening/closing windows.",
    "The system identifies activities including typing, opening envelopes, throwing garbage, picking fruit, and picking up items.",
    "Additional actions it recognizes include answering the phone, using a mouse, wiping the table, writing on the blackboard, and washing hands.",
    "It further detects behaviors like using a phone, reading, watering plants, walking to specific locations (e.g., bed, chair, cabinet, window, or blackboard), and general walking.",
    "Finally, the system captures motions such as getting out of bed, standing up, lying down, standing still, and lying still."
]

@register_model('TAD_single_1')
class TAD_single(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_single, self).__init__()
        self.config = config
        if config.embed_type == 'Norm':
            self.embedding = Embedding(config.in_channels, stride=config.embedding_stride)
        else:
            self.embedding = TADEmbedding(config.in_channels, out_channels=512, layer=3, input_length=config.input_length)

        logger.info(f'load {config.embed_type} embedding')
        logger.info(f'load {config.backbone_name}')
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
        self.embedding_model_name = "xlm-roberta-base"
        self.embedding_text = TextEmbeding(device="cuda:0", default_model=self.embedding_model_name)


    def fusion(self, text_features, wifi_features):
         # 元素级融合
        gated_text = 0.1 * text_features # (B, C, L)
        gated_wifi = 0.9 * wifi_features  # (B, C, L)

        combined_features = gated_text + gated_wifi

        return combined_features
    
    def forward(self, input):
        x = input[self.modality]
        B, C, L = x.size()

        x_text = self.embedding_text.cal_text_features_2d([labels], model_type=self.embedding_model_name, device="cuda:0")  # 其他的嵌入模型
        # x_text = self.embedding_text([labels], x.device).to(x.device)


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
