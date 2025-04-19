import json
import torch
import torch.nn as nn
from functools import partial
from model_gpt import mmCLIP_gpt_multi_brach_property_v3, ViT_wo_patch_embed

class CombinedModel(nn.Module):
    """A model combining CSI and text features with gated fusion."""
    
    # Model dimension mapping for different backbones
    MODEL_DIM_MAP = {
        "qwen_text-embedding-v3": 768,
        "llama": 4096,
        "clip-vit-large-patch14": 512,
        "t5-small": 512,
        "t5-base": 768,
        "xlm-roberta-base": 768
    }

    def __init__(self, csi_encode, is_text=False, embed_type="simple", model_key="llama", device="cuda:0"):
        """
        Initializes the CombinedModel.

        Args:
            csi_encode (nn.Module): CSI encoder module.
            is_text (bool): Whether to use text fusion. Default: False.
            embed_type (str): Type of text embedding to use. Default: 'simple'.
            model_key (str): Key for model dimension mapping. Default: 'llama'.
            device (str): Device for computations. Default: 'cuda:0'.
        """
        super().__init__()
        self.csi_encode = csi_encode
        self.embed_type = embed_type
        self.is_text = is_text
        self.device = torch.device(device)

        # Validate model key and set embed dimension
        if model_key not in self.MODEL_DIM_MAP:
            raise ValueError(f"Invalid model_key: {model_key}. Supported keys: {list(self.MODEL_DIM_MAP.keys())}")
        self.embed_dim = self.MODEL_DIM_MAP[model_key]

        # Initialize text self-attention module
        self.text_self_attention = ViT_wo_patch_embed(
            global_pool=False,
            embed_dim=self.embed_dim,
            depth=1,
            num_heads=4,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # Load text embeddings and initialize projection layer
        self.text_embeds = self._load_json_to_tensor(
            filename="./xrf55.json",
            target_key=embed_type,
            model_key=model_key
        )
        self.text_proj = nn.Linear(self.embed_dim, 55)

    @staticmethod
    def _load_json_to_tensor(filename: str, target_key: str, model_key: str, return_type: str = "tensor"):
        """
        Loads JSON data and returns the specified key's value as a tensor or raw data.

        Args:
            filename (str): Path to the JSON file.
            target_key (str): Key to extract from the JSON data.
            model_key (str): Model-specific key in the JSON data.
            return_type (str): Return type ('tensor' or 'raw'). Default: 'tensor'.

        Returns:
            torch.Tensor or Any: Tensor or raw value based on return_type.
        
        Raises:
            FileNotFoundError: If the JSON file does not exist.
            KeyError: If target_key or model_key is not found in the JSON data.
        """
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {filename}")

        if model_key not in data:
            raise KeyError(f"Model key '{model_key}' not found in JSON data")
        if target_key not in data[model_key]:
            raise KeyError(f"Target key '{target_key}' not found in {model_key} data")

        if return_type.lower() == "tensor":
            return torch.tensor(data[model_key][target_key], device="cuda:0")
        return data[model_key][target_key]

    def _cal_text_features_2d(self):
        """
        Processes text embeddings with self-attention and projection.

        Returns:
            torch.Tensor: Processed text features of shape [55].
        """
        text_embeds = self.text_embeds.to(self.device)

        # Ensure correct shape: [55, 1, embed_dim]
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(1)

        # Apply self-attention
        text_embeds_att = self.text_self_attention(text_embeds)  # [55, 1, embed_dim]
        text_embeds_att = text_embeds_att.unsqueeze(1)  # [55, 1, 1, embed_dim]

        # Concatenate original and attention-processed embeddings
        text_embeds = torch.cat([text_embeds.unsqueeze(1), text_embeds_att], dim=1)  # [55, 2, embed_dim]

        # Project and aggregate
        text_embeds = self.text_proj(text_embeds)  # [55, 2, 55]
        text_embeds = text_embeds.mean(dim=1).mean(dim=0)  # [55]

        return text_embeds

    def fusion(self, text_features, wifi_features):
        """
        Performs gated element-wise fusion of text and WiFi features.

        Args:
            text_features (torch.Tensor): Text features.
            wifi_features (torch.Tensor): WiFi features.

        Returns:
            torch.Tensor: Fused features.
        """
        gated_text = 0.1 * text_features
        gated_wifi = 0.9 * wifi_features
        return gated_text + gated_wifi

    def forward(self, inputs, labels=None):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor for CSI encoding.
            labels (Any, optional): Labels (not used in computation). Default: None.

        Returns:
            torch.Tensor: Model output (fused or CSI features).
        """
        csi_features = self.csi_encode(inputs)

        if self.is_text:
            text_features = self._cal_text_features_2d()
            csi_features = self.fusion(text_features, csi_features)

        return csi_features