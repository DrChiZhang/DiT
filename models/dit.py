# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=64, patch_size=2, in_chans=3, embed_dim=768):
#         super().__init__()
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)  # [B, C, H, W]
#         x = x.flatten(2).transpose(1, 2)  # [B, N, C]
#         return x

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
        
class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "hidden dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash Attention / Scaled Dot Product Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )

        # Merge heads
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
"""
    ┌────────────┐    fourier embedding          ┌──────────────┐
scalar    t │            │ ───────────────────────────→ │ vector [D]   │
            └────────────┘   (D = frequency_embed_size) └──────────────┘
"""
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations suitable for use as 
    conditional input in (for example) Diffusion Transformers (DiT), DDPM, etc.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # MLP to lift the frequency-encoded vector to model's hidden size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # Swish activation for added non-linearity
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # Length of the initial frequency (sin/cos) embedding
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings for conditional generation models.

        Args:
            t (torch.Tensor): 1-D Tensor of timesteps, shape [batch,] (can be fractional).
            dim (int):       Output embedding dimension (should be even for perfect pairs).
            max_period (float): Controls the minimum frequency/band in the embedding.

        Returns:
            torch.Tensor of shape [batch, embedding_dim]: sinusoidal timestep embeddings.
        """
        # Half the embedding dims will be cos, the rest sin
        half = dim // 2
        # Build exponentially-decaying frequencies for positional encoding, similar to Transformer
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # Outer product: (batch, 1) * (1, half) => (batch, half)
        args = t[:, None].float() * freqs[None]
        # Concatenate cos and sin of different frequencies as the final embedding
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # If dim is odd, pad with extra 0 for correct shape
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # Step 1: Compute sinusoidal frequency embedding for timesteps
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # Step 2: Project (and nonlinearly process) the frequency embedding to hidden vector
        t_emb = self.mlp(t_freq)
        # Output is [batch, hidden_size], suitable for injection into transformer/adaLN, etc.
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Handles conditional dropout for classifier-free guidance (CFG) in generative models,
    enabling both conditional and unconditional sampling.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # If dropout is used, add an extra embedding for "no label" (unconditional).
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Implements label dropout for classifier-free guidance:
        With dropout_prob, or according to the given mask, some labels are replaced
        by a special "unconditional" token (index = num_classes).
        
        Args:
            labels: [batch,] tensor of original class indices.
            force_drop_ids: Optional mask (bool/int tensor) where 1 marks labels to be dropped.
        Returns:
            labels: batch tensor, with some labels replaced by "unconditional".
        """
        if force_drop_ids is None:
            # Randomly drop labels with probability dropout_prob
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            # Use the externally provided drop mask
            drop_ids = force_drop_ids == 1
        # Replace dropped labels with index for "no label" (unconditional embedding)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """
        Args:
            labels:     [batch,] tensor of class ids.
            train:      Whether training (enables dropout if applicable)
            force_drop_ids: Optional mask to forcibly drop specified labels.

        Returns:
            embeddings: [batch, hidden_size] tensor of label embeddings.
        """
        # Only apply dropout during training (unless force_drop_ids is given)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # Lookup the embeddings (including possible "unconditional" label)
        embeddings = self.embedding_table(labels)
        return embeddings

class DiTBlock(nn.Module):
    """
    A DiT (Diffusion Transformer) block with adaptive layer norm zero (adaLN-Zero) conditioning.
    This enables conditional control (e.g., timestep embedding, class, text, etc.) to modulate
    both self-attention and MLP branches via shift, scale, and gating.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # LayerNorm layers; no affine as modulation will be performed conditionally
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Multi-head self-attention
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # MLP hidden dimension, typically 4x the hidden size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Use GELU activation with tanh approximation (faster)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # MLP block (typically two linear layers and nonlinearity)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # adaLN_modulation: maps conditioning vector c to six modulation vectors
        # Used for (shift, scale, gate) for both attention and MLP branches
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),                                       # nonlinearity
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)  # to 6*hidden for chunking
        )

    def forward(self, x, c):
        # c: the conditioning vector (e.g., timestep encoding)
        # Pass c through the modulation head; chunk into
        # [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp], each [batch, hidden]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # --- Attention branch ---
        # 1. Normalize x, modulate using conditioning vars (shift & scale),
        # 2. Feed through self-attention
        # 3. Gate and add as residual
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # --- MLP branch ---
        # Same logic as above: LayerNorm -> modulate -> MLP -> gate -> residual add
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        
        # Return modulated & residual-updated output
        return x


class FinalLayer(nn.Module):
    """
    The final (output) layer of the DiT (Diffusion Transformer) model.
    This layer adapts features for conditional image synthesis.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # Final LayerNorm (without affine parameters, as conditional modulation will be applied)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Linear head: maps token features to pixel/patch output (e.g., 16x16 patch * 3 channels)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # adaLN_modulation: generates conditional shift and scale for modulation from condition c
        # Input: conditioning vector of shape [batch, hidden_size]
        # Output: [shift, scale], each of shape [batch, hidden_size]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # non-linear activation (Swish)
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)  # project to 2 * hidden_size
        )

    def forward(self, x, c):
        """
        x: [batch, num_patches, hidden_size] - Output tokens from DiT/Transformer
        c: [batch, hidden_size] - Conditional embedding (e.g., timestep, class, text)
        """
        # Use conditioning to produce (shift, scale) vectors (for adaptive LayerNorm modulation)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # Apply LayerNorm, then modulate with (shift, scale) in a feature-wise affine manner
        # (modulate: usually x * (1 + scale) + shift, per token channel)
        x = modulate(self.norm_final(x), shift, scale)
        # Map feature tokens to image patch pixels/channels via linear head
        x = self.linear(x)
        # Output: typically [batch, num_patches, patch_size * patch_size * out_channels]
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    Supports conditional generation (timestep, label) via adaLN-Zero modulation.
    """
    def __init__(
        self,
        patch_size=2,
        num_classes=1000,
        input_size=32,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Patch embedding layer: splits image into patches and projects to hidden_size, i.e. embed_dim
        """
        Input: [B, C, H, W]
        Output: [B, N_patches, embed_dim]
        """
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # Timestep embedding (for diffusion step conditioning)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Class label embedding (for CFG and class-conditional generation)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # Fixed (frozen) 2D sin-cos positional embeddings (not learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Stack of DiTBlocks (transformer backbone)
        """
            Input x
            │
            [Block1]──┬───────────┐
            │      │           :
            [Block2]──┴──...────>[Block N]
                    │
                    Output
        """
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # Output head
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # Specialized weight initialization throughout backbone
        self.initialize_weights()

    def initialize_weights(self):
        """
        Applies initialization to all components in the network: transformers, embeddings, etc.
        - Xavier for linear/proj layers.
        - Normal for embedding tables.
        - Zeroes certain layers for adaLN-Zero behavior.
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Setup fixed pos_embed using 2d sinusoidal
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # Patch embedding (flattened linear init for Conv2d context)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Set all adaLN-Zero modulation layers (DiT blocks and output) to zero (learn only deviation)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Output layer (final patch pixel head) zeroed out, so model starts as identity
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Rearranges model outputs from patch tokens back to image space.
        x: (B, T, patch_size**2 * C) - PATCHED
        imgs: (B, C, H, W) - FULL IMAGE
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, y):
        """
        DiT 模型的前向传播函数。

        参数：
            x : Tensor，形状 (N, C, H, W)
                输入图像或图像的潜在表示（通常是扩散过程中某一步的 x_t）
            t : Tensor，形状 (N,)
                当前的扩散 timestep（整数）。用于添加时间条件。
            y : Tensor，形状 (N,)
                条件标签（例如分类标签），用于有条件生成。

        返回：
            Tensor，形状 (N, out_channels, H, W)
            对 x_t 的预测结果（可能是噪声 ε 或 x_0 等），
            用于与真实值做损失比较或在采样过程中作为中间状态。
        """
        # -------- 1. 对输入图像进行 patch 分割和线性投影 --------
        # self.x_embedder 是一个 PatchEmbedding 模块（例如 Conv2d 或 linear projection + reshape）
        # 它会将图像 x 分割成若干个 patch，然后将每个 patch 映射到 D 维 embedding 空间
        #
        # pos_embed 是 learnable 的位置编码，表示每个 patch 的空间位置
        # 它会与每个 patch embedding 相加，使模型获得空间结构感知能力
        #
        # 输出 x 的 shape 是 (N, T, D)，其中：
        #   - N 是 batch size
        #   - T 是 patch 个数（通常为 H×W / patch_size²）
        #   - D 是 token 的维度（Transformer 的 embedding size）
        x = self.x_embedder(x) + self.pos_embed
        # -------- 2. 时间步嵌入 --------
        # self.t_embedder 将 timestep t（整数）映射为 D 维向量
        # 通常是一个 TimeEmbedding 模块（如 sinusoidal 或 learnable embedding + MLP）
        #
        # 输出 t 的 shape 是 (N, D)，表示每个样本对应的时间信息（扩散程度）
        t = self.t_embedder(t)
        # -------- 3. 类别嵌入（条件标签） --------
        # self.y_embedder 将类别标签 y 映射为 D 维向量
        # 在训练过程中，y_embedder 可能会随机将一部分 y 设置为 None（实现 CFG 的无条件部分）
        #
        # 输出 y 的 shape 也是 (N, D)，与 t 相同
        y = self.y_embedder(y, self.training)
        # -------- 4. 合并时间嵌入和类别嵌入作为上下文条件 --------
        # 将时间 t 和类别 y 相加，得到一个统一的条件向量 c
        # 这个条件会传递给 transformer 的每一层 block，用于调节每层的输出
        c = t + y

        # -------- 5. Transformer Block 前向传播 --------
        # 遍历多层 transformer block，每一层都对 token 序列进行自注意力和前馈网络处理
        # 每个 block 同时接受：
        #   - x: 当前 token 序列（图像 patch 的嵌入）
        #   - c: 条件向量，用于调节注意力或添加到 embedding 中（如 FiLM、cross-attn 等）
        #
        # 输出 x 的形状仍保持不变：(N, T, D)
        for block in self.blocks:
            x = block(x, c)

        # -------- 6. Transformer 最后一层输出映射 --------
        # 将 transformer 最后的输出 token 序列，转换为 patch 的像素值
        # 也就是从 token embedding 空间 (D) 映射回 patch 空间
        #
        # 通常每个 token 输出 patch_size² × out_channels 维的向量，
        # 用于恢复一个 patch 的像素块
        # (N, T, patch_size ** 2 * out_channels)
        x = self.final_layer(x, c)                

        # -------- 7. 将 patch token 重构为图像 --------
        # 使用 self.unpatchify 将 token 序列拼接还原为图像：
        #   - 按顺序将 patch 排布回 H×W 的图像结构
        #   - 通道数设为 out_channels（如 RGB 的 3 或 latent 的 4）
        #
        # 通常会 reshape + permute + reshape 来实现
        x = self.unpatchify(x)                 

        # (N, C, H, W) or (N, C*2, H, W) if learn_sigma is true. 
        return x
    

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        利用 Classifier-Free Guidance（CFG）机制执行前向传播。
        该机制在采样过程中运行有条件与无条件两个前向预测，并通过 cfg_scale 控制其插值强度，
        以提升生成图像对条件（如文本、类别等）的响应能力。

        参数：
            x : Tensor
                当前的扩散图像 x_t，形状通常为 [B, C, H, W]
            t : Tensor
                当前扩散的时间步 step（例如 DDPM 中的 t），通常为形如 [B] 的整数张量
            y : 条件信息（如类别标签、文本 embedding）
            cfg_scale : float
                指导强度（scale）。通常为 1.0（不引导）或 >1.0（增强对条件的遵循）

        返回：
            Tensor，预测输出，结构与模型原始输出一致（通常为预测噪声和可能的方差信息）
        """
        
        # Step 1: 将输入 x 分为一半，用于创建一组无条件和有条件的输入
        # x[:len(x)//2] 相当于 batch 的前半部分，假设 batch size 为 8，则取前 4 个样本
        half = x[: len(x) // 2]  # 形状：[B/2, C, H, W]

        # Step 2: 将这 4 个样本复制一遍，拼接成 [8, C, H, W]，用于分别进行无条件与有条件预测
        # 第一个 4 个用于无条件预测（如 y=None），后 4 个用于有条件预测（如 y=文本、标签）
        combined = torch.cat([half, half], dim=0)

        # Step 3: 运行模型前向传播，传入 combined, t, y
        # 由于 combined 有 2 份，y 也必须能与其匹配（可能在 self.forward 内处理 y=None 的情况）
        model_out = self.forward(combined, t, y)  # 形状：[B, C_out, H, W]

        # Step 4: 拆分模型输出为两个部分
        # - 第一部分是预测的 ε（噪声分量），一般用于还原 x0 或做去噪
        # - 第二部分可能是预测的 log-variance 或其他辅助分量（如 DDIM/GuidedDiffusion 结构）

        # 可选：使用下面一行替换当前这行，可以对所有通道进行 CFG（标准做法）
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]

        # 这里按 OpenAI GLIDE 的实现，只对前三个通道做 CFG（例如 RGB）
        eps, rest = model_out[:, :3], model_out[:, 3:]

        # Step 5: 拆分 eps 为两个部分：有条件的预测和无条件的预测
        # - 前半部分是无条件预测结果（通常是 y=None）
        # - 后半部分是有条件预测结果（使用 y）
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        # Step 6: 使用 Classifier-Free Guidance 公式插值
        #   ε_guided = ε_uncond + scale * (ε_cond - ε_uncond)
        #   - 当 scale = 1：结果等于无条件预测，不引导
        #   - 当 scale > 1：结果更偏向于条件预测，可提升生成样本质量
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        # Step 7: 将插值结果拼接回完整 batch（复制一份是因为 batch 对应了两种情况）
        eps = torch.cat([half_eps, half_eps], dim=0)

        # Step 8: 将经过引导的 eps 与剩余部分（如 log-variance）重新拼接，作为模型最终输出
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sin-cos positional embeddings for grid patches.

    Args:
        embed_dim (int):      The embedding dimension for each token.
        grid_size (int):      The number of patches per grid side (image assumed square).
        cls_token (bool):     Whether to prepend extra tokens (e.g., class token).
        extra_tokens (int):   Number of extra tokens; position embeddings for these will be zero vectors.

    Returns:
        pos_embed (np.ndarray): [grid_size*grid_size, embed_dim] or 
                                [extra_tokens+grid_size*grid_size, embed_dim]
                                (depending on cls_token/extra_tokens)
    """
    # Create grid coordinates for each patch (y and x indices)
    grid_h = np.arange(grid_size, dtype=np.float32)                # Row indices
    grid_w = np.arange(grid_size, dtype=np.float32)                # Column indices
    grid = np.meshgrid(grid_w, grid_h)                             # Create coordinate matrices (x goes first)
    grid = np.stack(grid, axis=0)                                  # Stack to shape [2, grid_size, grid_size]

    # Reshape to [2, 1, grid_size, grid_size] for downstream processing
    grid = grid.reshape([2, 1, grid_size, grid_size])
    # Compute the actual positional embedding using sin and cos on the coordinates
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid) # shape: [grid_size * grid_size, embed_dim]

    # Optionally, add zero-vector embedding(s) for class/distillation tokens at the beginning
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    # Returned shape: [num_tokens, embed_dim], where num_tokens = grid_size*grid_size (+ extra_tokens if used)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sine-cosine positional embeddings from a 2D spatial grid.

    Args:
        embed_dim (int):   The embedding dimension to generate (must be even).
        grid (np.ndarray): The spatial grid of patch coordinates, shape [2, H, W]

    Returns:
        emb (np.ndarray):  Positional embeddings, shape [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0

    # Use the first half of the embedding dimensions to encode y-coordinates (height)
    # grid[0]: y (or h) coordinates, shape [H, W]
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)

    # Use the second half for x-coordinates (width)
    # grid[1]: x (or w) coordinates, shape [H, W]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # Concatenate the embeddings for y and x axes to get the full embedding for each patch
    # Resulting shape: [H*W, embed_dim]
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sine-cosine positional embeddings for a set of positions.

    Args:
        embed_dim (int): Output dimension for each embedding (must be even).
        pos (np.ndarray): Positions to be encoded, shape (M,)
    
    Returns:
        emb (np.ndarray): 1D sin-cos positional embeddings, shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0

    # Create the frequency coefficients (omega) for the encoding.
    # This establishes a geometric progression of frequencies from 1/10000^0 up to 1/10000^1.
    omega = np.arange(embed_dim // 2, dtype=np.float64)       # [0, 1, ..., D/2 - 1]
    omega /= embed_dim / 2.                                   # normalize axis to [0, 1)
    omega = 1. / (10000 ** omega)                             # [1, 1/10000^(1/(D/2)), ..., 1/10000]

    # Reshape position vector to (M,) and compute outer product with omega (results in all pairwise combinations)
    pos = pos.reshape(-1)                                     # Ensure shape (M,)
    out = np.einsum('m,d->md', pos, omega)                    # (M, D/2): for each pos, all frequencies

    # Compute sine and cosine of the resulting matrix, for each position and frequency
    emb_sin = np.sin(out)                                     # (M, D/2)
    emb_cos = np.cos(out)                                     # (M, D/2)

    # Concatenate sin and cos along the last axis, resulting in full positional encoding
    emb = np.concatenate([emb_sin, emb_cos], axis=1)          # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
