import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats: int):
        super().__init__()
        self.row_embed = nn.Embedding(256, num_pos_feats)
        self.col_embed = nn.Embedding(256, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1)
        ], dim=-1)  # (H,W,2F)
        pos = pos.permute(2,0,1).unsqueeze(0).repeat(B,1,1,1)  # (B,2F,H,W)
        if pos.shape[1] != C:
            proj = nn.Conv2d(pos.shape[1], C, 1).to(x.device)
            pos = proj(pos)
        return pos

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B,C,H,W = x.size()
        q = self.query_conv(x).view(B, -1, H*W).permute(0,2,1)
        k = self.key_conv(x).view(B, -1, H*W)
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        v = self.value_conv(x).view(B, -1, H*W)
        out = torch.bmm(v, attn.permute(0,2,1)).view(B,C,H,W)
        return self.gamma * out + x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.qkv = nn.Conv2d(dim, dim*3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B,C,H,W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def reshape(t):
            t = t.view(B, self.num_heads, self.head_dim, H*W)
            return t.permute(0,1,3,2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        attn = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0,1,3,2).contiguous().view(B,C,H,W)
        return self.proj(out)
