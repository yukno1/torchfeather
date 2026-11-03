import math

import torch

from torchfeather.model.model_args import DeepSeekV3ModelArgs


def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    """
    RoPE 独立应用与每个头。在 MLA 的特定情况下，会有不同。
    多头 attention 中， 每个头有一个特定的小 embedding 向量。
    将 RoPE 用在每个 embedding 向量
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seg_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len
    ) -> float:
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.floor(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # basic RoPE frequency calculation
    # 旋转每个维度，旋转角度取决于位置
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN scaling for extended context
    # YaRN is used to extent the context length after pre-training
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # create position indices
    # 训练期间所有可能的位置
    t = torch.arange(seqlen)

    # outer product: [positions] x [frequencies]
    # 位置和频率的所有可能组合，为每个 m 和 theta_i
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i * freqs * pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # freqs_shape: [max_seq_len, D/2]
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    会得到一批向量
    """
    # x: [Batch, Seqlen, Headdim, Dim]
    dtype = x.dtype
    # x => [B, S, H, D/2, 2]， 每组两个维度，视为复数。x 是想要旋转的向量
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))  # => [B, S, H, D/2]
    # freqs_cis 与 x 形状不兼容，需要进行广播
    # freq_cis: [S, D/2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))  # => [1, S, 1, D/2]
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)
