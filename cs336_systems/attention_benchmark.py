import torch
from einops import einsum
from torch import Tensor
from jaxtyping import Float, Bool
import math


d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384]


def softmax(in_features: Float[Tensor, " ..."], dim: int):
    # Subtract max value
    repeat_vec = [1 for _ in range(len(in_features.shape))]
    repeat_vec[dim] = in_features.shape[dim]
    max_v, _ = in_features.max(dim, keepdim=True)
    max_v = max_v.repeat(*repeat_vec)

    # Calculate exp and the sum
    ev = (in_features - max_v).exp()
    sum_ev = ev.sum(dim, keepdim=True)
    sum_ev = sum_ev.repeat(*repeat_vec)

    return ev / sum_ev


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None
) -> Float[Tensor, " ... queries d_v"]:
    # queries, keys, and values == seq_len
    # d_v == d_k
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        # qk[~mask] = -torch.inf # forward works, but not autograde
        qk = qk.masked_fill(~mask, float("-inf"))

    # softmax over keys
    sm = softmax(qk, -1)
    return einsum(sm, V, "... queries keys, ... keys d_v -> ... queries d_v")

forward_passes = 100
batch_size = 8

for d_model in d_models:
    for seq_len in seq_lens:
        used_bytes = []
         # Create mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"))
        full_mask = mask.unsqueeze(0).unsqueeze(0)

        for p in range(forward_passes):
            torch.cuda.synchronize()
            alloc_bytes0 = torch.cuda.memory_allocated("cuda")

            Q = torch.randn((batch_size, seq_len, d_model), device="cuda")
            K = torch.randn((batch_size, seq_len, d_model), device="cuda")
            V = torch.randn((batch_size, seq_len, d_model), device="cuda")

            qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_model)
            qk = qk.masked_fill(~mask, float("-inf"))

            # softmax over keys
            sm = softmax(qk, -1)
            attn = einsum(sm, V, "... queries keys, ... keys d_v -> ... queries d_v")

            torch.cuda.synchronize()

            alloc_bytes1 = torch.cuda.memory_allocated("cuda")
            used_bytes.append(alloc_bytes1 - alloc_bytes0)

            # free memory
            del(Q)
            del(K)
            del(V)
            del(qk)
            del(sm)
            del(attn)

        print(f"d_model {d_model}, seq_len {seq_len}, used bytes {torch.tensor(used_bytes, dtype=torch.float).mean()}")
