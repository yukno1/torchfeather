from dataclasses import dataclass, field

from torch import nn
from loguru import logger

from torchfeather.model.moe import MoEArgs


@dataclass
class DeepSeekV3ModelArgs:
    max_seg_len: int = 4096 * 4
    vocab_size: int = 102400
    dim: int = 2048  # size of embedding vector
    inter_dim: int = 10944  # size of ffn projection
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5  # eps used for RMSNorm

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    # Multi-Head Latent Attention (MLA)
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_experts = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                # 嵌入层参数
                nparams_embedding += p.numel()
                nparams_dense += p.numel()  # 嵌入层属于稠密参数
            elif "moe.shared_experts" in name:
                nparams_shared_experts += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            else:
                # 否则一定是稠密模型
                nparams_dense += p.numel()

        nparams_sparse = (
            nparams_moe_router + nparams_shared_experts + nparams_experts
        )  # 所有属于MoE模型的参数
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_experts
            + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
        )

        logger.info(
            f"Total parameter count: dense {nparams_dense:,}"
            f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
        )

        n_layers = self.n_layers
        n_heads = self.n_heads
        head_dims = self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim

        num_flops_per_token = (
            6
            * (
                nparams_dense  # 全部非 MoE 参数
                - nparams_embedding  # 嵌入本质只是查字典
                + nparams_sparse_active  # 全部 MoE 参数
            )
            + 6 * n_layers * n_heads * head_dims * seq_len  # 注意力的参数
        )  # 经验法则：每个token需要的flops = 6 * P

        return nparams, num_flops_per_token
