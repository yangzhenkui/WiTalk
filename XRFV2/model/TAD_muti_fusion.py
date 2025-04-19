import torch
import torch.nn as nn
import logging
from model.models import register_model, make_backbone, make_backbone_config
from model.head import ClsLocHead
from model.TAD.embedding import Embedding
from model.model_config import TAD_single_Config
from model.embedding import TADEmbedding, TADEmbedding_pure, NoneEmbedding, WifiEmbeding
from model.fusion import GatedFusion, GatedFusionAdd, GatedFusionWeight, GatedFusionAdd2
from model.TextEmbeding import TextEmbeding
# from model_gpt import mmCLIP_gpt_multi_brach_property_v3


label = ["The system can recognize actions such as stretching, pouring water, writing, cutting fruit, eating fruit, and taking medicine.",
"It also detects actions like drinking water, sitting down, turning on/off the eye protection lamp, and opening/closing curtains.",
"The system can identify activities such as opening/closing windows, typing, opening envelopes, throwing garbage, and picking fruit.",
"Other actions include picking up items, answering the phone, using a mouse, wiping the table, and writing on the blackboard.",
"It also recognizes actions like washing hands, using a phone, reading, watering plants, and walking to different locations (e.g., bed, chair, cabinet, window, blackboard).",
"Finally, the system can detect movements like getting out of bed, standing up, lying down, standing still, and lying still."]




logger = logging.getLogger(__name__)

@register_model('TAD_muti_weight_grc')
class TAD_muti_weight_grc(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_weight_grc, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)
        # self.embedding_text = TextEmbedding()

        # self.fusion = GatedFusionWeight(hidden_size=config.out_channels)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)  # 2048 -> 256

        self.embedding = TADEmbedding_pure(config.out_channels, out_channels=512, layer=3, input_length=config.input_length)
        self.embedding_model_name = "t5-small"
        self.embedding_text = TextEmbeding(device="cuda:0", default_model=self.embedding_model_name)

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

    def fusion(self, text_features, wifi_features):
         # 元素级融合
        gated_text = 0.2 * text_features # (B, C, L)
        gated_wifi = 0.8 * wifi_features  # (B, C, L)

        combined_features = gated_text + gated_wifi

        return combined_features

    def forward(self, input, labels):
        
        # x_imu = input['imu']
        x_wifi = input['wifi']
        # B, C, L = x_imu.size()
        B, C, L = x_wifi.size()
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 30, 2048]) torch.Size([4, 270, 2048])
        # x_imu = self.embedding_imu(x_imu)
        x_text = self.t5_base.cal_text_features_2d([labels], model_type=self.embedding_model_name, device="cuda:0")
        # x_text = self.embedding_text([labels])
        # print(x_text.shape)
        x_wifi = self.embedding_wifi(x_wifi)
        # print(x_imu.shape, x_wifi.shape)  // [B, 512, 2048]
        # x = self.fusion(x_imu, x_wifi)
        x = self.fusion(x_text, x_wifi)

        x = self.embedding(x)  # TSSE编码
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