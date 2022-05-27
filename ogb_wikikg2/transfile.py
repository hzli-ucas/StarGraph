import torch
from torch import nn

class DropPath(nn.Module):
    """Drop path

    Randomly drop the input (i.e., output zero) with some probability, per sample.
    """

    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, x):

        if self.dropout_p == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.dropout_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0 / 1

        # Discussion: https://github.com/rwightman/pytorch-image-models/discussions/895
        output = x / keep_prob * random_tensor
        return output


class MLP(nn.Module):
    """MLP layer, usually used in Transformer"""

    def __init__(
        self,
        in_feat: int,
        mlp_ratio: int = 1,
        out_feat: int = 0,
        dropout_p: float = 0.0,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        mid_feat = in_feat * mlp_ratio
        out_feat = out_feat or in_feat

        self.act = act_layer()

        self.linear1 = nn.Linear(in_feat, mid_feat)
        self.drop1 = nn.Dropout(dropout_p)

        self.linear2 = nn.Linear(mid_feat, out_feat)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.drop1(self.act(self.linear1(x)))
        x = self.drop2(self.linear2(x))
        return x


class AttentionLayer(nn.Module):
    """Multi-head scaled self-attension layer"""

    def __init__(self, num_feat: int, num_heads: int = 8, qkv_bias: bool = False, dropout_p: float = 0.0):
        super().__init__()

        assert num_feat % num_heads == 0

        self.num_feat = num_feat
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.num_feat, self.num_feat * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_p)

        self.proj = nn.Linear(self.num_feat, self.num_feat)
        self.proj_drop = nn.Dropout(dropout_p)

    def forward(self, x):
        B, L, C = x.shape
        assert C == self.num_feat

        qkv = self.qkv(x)  # [B, L, num_feat * 3]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # [B, L, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv.unbind(0)  # [B, num_heads, L, head_dim] * 3

        attn = q @ k.transpose(-2, -1)  # [B, num_heads, L, L]
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)  # [B, L, num_heads, head_dim]
        x = x.reshape(B, L, self.num_feat)  # [B, L, num_feat]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_feat: int,
        out_feat: int = 0,
        num_heads: int = 8,
        qkv_bias: bool = False,
        mlp_ratio: int = 4,
        dropout_p: float = 0.0,
        droppath_p: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        out_feat = out_feat or in_feat

        self.droppath = DropPath(droppath_p)

        self.norm1 = norm_layer(in_feat)
        self.norm2 = norm_layer(in_feat)

        self.attn = AttentionLayer(num_feat=in_feat, num_heads=num_heads, qkv_bias=qkv_bias, dropout_p=dropout_p)

        self.mlp = MLP(
            in_feat=in_feat, mlp_ratio=mlp_ratio, out_feat=out_feat, dropout_p=dropout_p, act_layer=act_layer
        )

    def forward(self, x):
        x = x + self.droppath(self.attn(self.norm1(x)))
        x = x + self.droppath(self.mlp(self.norm2(x)))
        # x = self.droppath(self.mlp(self.norm2(x)))
        return x