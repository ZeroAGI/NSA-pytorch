import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NativeSparseAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, comp_block=32, sel_block=32, win_size=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.comp_block = comp_block
        self.sel_block = sel_block
        self.win_size = win_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.comp_mlp = nn.Sequential(
            nn.Linear(comp_block*self.head_dim, self.head_dim//2),
            nn.ReLU(),
            nn.Linear(self.head_dim//2, self.head_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(self.head_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, L, E = x.shape  # Batch size, sequence length, embedding dimension
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, E/H)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, E/H)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, E/H)

        # Compressed attention
        k_compressed = self._compress(k, self.comp_block)  # Compress keys
        v_compressed = self._compress(v, self.comp_block)  # Compress values
        attn_compressed = self._scaled_dot_product_attention(q, k_compressed, v_compressed)

        comp_attn = self._comp_attn(q, k_compressed)

        # Selected attention
        k_selected, v_selected = self._select(k, v, comp_attn)
        attn_selected = self._scaled_dot_product_attention(q, k_selected, v_selected)

        # Sliding window attention
        k_window = k[:, :, -self.win_size:]  # Take the last win_size tokens
        v_window = v[:, :, -self.win_size:]
        attn_window = self._scaled_dot_product_attention(q, k_window, v_window)

        # Combine attention outputs using gating mechanism
        gate_weights = self.gate(q)
        attn_output = gate_weights[:, :, :, 0].unsqueeze(-1) * attn_compressed + \
                      gate_weights[:, :, :, 1].unsqueeze(-1) * attn_selected + \
                      gate_weights[:, :, :, 2].unsqueeze(-1) * attn_window

        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.num_heads*self.head_dim)

        return attn_output

    def _comp_attn(self, q, k):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights

    def _scaled_dot_product_attention(self, q, k, v):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

    def _compress(self, x, comp_block):
        # Compress keys/values by aggregating blocks
        B, H, L, E = x.shape
        x_blocks = x.view(B, H, -1, comp_block, E)
        x_compressed = self.comp_mlp(x_blocks.flatten(3))
        return x_compressed

    def _select(self, k, v, comp_attn):
        # Select important blocks based on attention scores
        B, H, L, E = k.shape

        k_blocks = k.view(B, H, -1, self.sel_block, E)
        v_blocks = v.view(B, H, -1, self.sel_block, E)

        block_scores = comp_attn.sum(dim=2)  # B, H, L_compressed
        topk_scores, topk_indices = torch.topk(
            block_scores,
            k=16,
            dim=-1
        )

        k_selected = torch.gather(k_blocks, dim=2, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k_blocks.shape[-2], k_blocks.shape[-1])).view(B, H, -1, E)
        v_selected = torch.gather(v_blocks, dim=2, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, v_blocks.shape[-2], v_blocks.shape[-1])).view(B, H, -1, E)

        return k_selected, v_selected


if __name__ == '__main__':
    # test
    batch_size = 2
    seq_length = 6400
    embedding_dim = 2560
    num_heads = 8 

    input_tensor = torch.randn(batch_size, seq_length, embedding_dim)
    print(f"Input shape: {input_tensor.shape}")
    nsa_module = NativeSparseAttention(embed_dim=embedding_dim, num_heads=num_heads)
    output = nsa_module(input_tensor)
    print(f"Output shape: {output.shape}")
