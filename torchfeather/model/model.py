import math
from typing import Optional

import torch
from torch import nn

from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.attention import ScaledDotProductAttentionWrapper
from torchfeather.model.moe import FeedForward, MoE
from torchfeather.model.rope import apply_rotary_emb, precompute_freqs_cis


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers
    """

    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers  # 前 n 层是稠密的
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        # This is different from GPT-2 style initialization, as visible in the HF
        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass for Transformer block

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input
        """
        x = x + self.attention(self.attention_norm(x), freqs_cis)  # 残差连接
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(
        self, init_std: float | None, buffer_device: torch.device | None = None
    ):
        if buffer_device is None:
            raise ValueError(
                "buffer_device must be provided for TransformerBlock weight initialization"
            )
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(
                init_std=self.weight_init_std, buffer_device=buffer_device
            )
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        )

        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)  # 输出归一化层
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype,
            bias=False,
        )  # 输出 MLP 层

    def init_weights(
        self, init_std: float | None, buffer_device: torch.device | None = None
    ):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        # embedding 层通常初始化为正态分布
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std, buffer_device)  # ty: ignore[call-non-callable]
        if self.norm is not None:
            self.norm.reset_parameters()
        # 为什么用标准差初始化输出层？
        # 为什么选择这个标准差？
        # 方差大：激活值爆炸->大损失，梯度爆炸；方差小：收缩，激活值消失
        final_out_std = self.model_args.dim ** (-0.5)
        cutoff_factor = 3
        if self.output is not None:
            # 从标准差采样，限制在 ±3σ
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for Transformer model

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this wiil be the input token indices for the ranks on the first pipeline stage.
                This will be the activation

        Returns:
            torch.Tensor: logits tensor of shape (batch_sime, vocab_size)
        """

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        # 实际上总是有 norm，embedding，output
        return output
