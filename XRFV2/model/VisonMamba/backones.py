import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba

class Block(nn.Module):
    """ Mamba 块，支持下采样，输入输出格式为 (B, D, L) """
    def __init__(self, dim, d_state=16, n_ds_stride=1, drop_path=0.):
        super().__init__()
        self.mixer = Mamba(d_model=dim, d_state=d_state, bimamba_type="v2")
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_ds_stride = n_ds_stride

    def forward(self, hidden_states, residual=None):
        # 输入: (B, D, L)
        # 转换为 Mamba 期望的 (B, L, D)
        hidden_states = hidden_states.transpose(1, 2)  # (B, L, D)
        if residual is not None:
            residual = residual.transpose(1, 2)  # (B, L, D)

        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        if self.n_ds_stride > 1:
            hidden_states = hidden_states[:, ::self.n_ds_stride, :]
            residual = residual[:, ::self.n_ds_stride, :]

        # 转换回 (B, D, L)
        hidden_states = hidden_states.transpose(1, 2)  # (B, D, L)
        residual = residual.transpose(1, 2)  # (B, D, L)
        return hidden_states, residual

class VisionMambaBackbone(nn.Module):
    def __init__(self, 
                 n_in=512, 
                 n_embd=512, 
                 n_embd_ks=3, 
                 arch=(2, 2, 5), 
                 scale_factor=2, 
                 with_ln=True, 
                 d_state=16):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)

        # 嵌入层，保持 (B, D, L) 格式
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(nn.Conv1d(in_channels, n_embd, kernel_size=n_embd_ks, 
                                      stride=1, padding=n_embd_ks//2))
            self.embd_norm.append(nn.LayerNorm(n_embd) if with_ln else nn.Identity())

        # Stem 层
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(Block(dim=n_embd, d_state=d_state))

        # Branch 层（带下采样）
        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(Block(dim=n_embd, d_state=d_state, n_ds_stride=scale_factor))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, mask=None):
        # x: (B, C, L), e.g., (16, 512, 256)
        # mask: (B, 1, L) or None
        B, C, L = x.shape
        if mask is None:
            mask = torch.ones(B, 1, L, device=x.device)

        # 嵌入层，保持 (B, D, L)
        for idx in range(len(self.embd)):
            x = self.embd[idx](x)
            # 应用 LayerNorm 时，转换到 (B, L, D) 再转回
            x = self.relu(self.embd_norm[idx](x.transpose(1, 2)).transpose(1, 2))

        # Stem 层
        residual = None
        hidden_states = x  # (B, D, L)
        for layer in self.stem:
            hidden_states, residual = layer(hidden_states, residual)

        # 输出准备
        out_feats = [hidden_states]
        out_masks = [mask]

        # Branch 层（下采样）
        for layer in self.branch:
            hidden_states, residual = layer(hidden_states, residual)
            mask = mask[:, :, ::self.scale_factor]
            out_feats.append(hidden_states)
            out_masks.append(mask)

        return tuple(out_feats), tuple(out_masks)

# 测试代码
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# model = VisionMambaBackbone(n_in=512, n_embd=512, arch=(2, 2, 5)).to(device)
# x = torch.randn(16, 512, 256).to(device)
# feats, masks = model(x)
# for i, feat in enumerate(feats):
#     print(f"Scale {i}: {feat.shape}")