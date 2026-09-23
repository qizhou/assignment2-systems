from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from einops import einsum


def flash_attention_single_batch(Q, K, V):
    N_q, d_model = Q.shape
    N_k, _ = K.shape

    # block sizes
    B_q = 16
    B_k = 16

    T_q = N_q // B_q
    T_k = N_k // B_k

    O = torch.zeros((N_q, d_model))
    L = torch.zeros((N_q,))

    for i in range(T_q):
        # load Qi from global memory
        Qi = Q[i*B_q:(i+1)*B_q]
        Oi = torch.zeros((B_q, d_model))
        li = torch.zeros((B_q,))
        mi = torch.full((B_q,), -torch.inf)

        for j in range(T_k):
            # load Kj, Vj from global memory
            Kj = K[j*B_k:(j+1)*B_k] # shape (B_k, d_model)
            Vj = V[j*B_k:(j+1)*B_k] # shape (B_k, d_model)

            Sij = einsum(Qi, Kj, "B_q d_model, B_k d_model -> B_q B_k") / math.sqrt(d_model) # shape (B_q, B_k)

            mi_new = torch.stack((mi, Sij.max(dim=-1).values)).max(dim=0).values # shape (B_q)
            # P = (Sij - mi_new.reshape((B_q, 1)).expand(B_q, B_k)).exp() # shape (B_q, B_k)
            P = (Sij - mi_new.reshape((B_q, 1))).exp() # shape (B_q, B_k)

            alpha = (mi - mi_new).exp() # shape (B_q)

            li_new = alpha * li + P.sum(dim=-1) # shape (B_q)

            # Oi = (mi - mi_new).exp().diag() @ Oi + P @ Vj # shape (B_q, d_model)
            Oi = alpha.reshape((B_q, 1)) * Oi + P @ Vj # shape (B_q, d_model)

            mi = mi_new
            li = li_new

        # O[i*B_q:(i+1)*B_q] = (1 / li).diag() @ Oi
        O[i*B_q:(i+1)*B_q] = (1 / li).reshape((B_q, 1)) * Oi
        L[i*B_q:(i+1)*B_q] = mi + li.log()

    return O, L


def attention_backward(Q, K, V, O, dO, L, is_causal=False):
    # Q shape (B, N_q, d_model)
    # K, V shape (B, N_k, d_model)
    # dO shape (B, N_q, d_model)
    B, N_q, d_model = Q.shape
    _, N_k, _ = K.shape
    S = einsum(Q, K, "... N_q d_model, ... N_k d_model -> ... N_q N_k") / math.sqrt(d_model) # shape (B, N_q, N_k)
    # P = (S - L.reshape(B, N_q, 1).expand(B, N_q, N_k)).exp() # shape (B, N_q, N_k)
    P = (S - L.reshape(B, N_q, 1)).exp() # shape (B, N_q, N_k)
    dV = einsum(P.transpose(1, 2), dO.transpose(1, 2), "... N_k N_q, ... d_model N_q -> ... N_k d_model") # shape (B, N_k, d_model)
    dP = einsum(dO, V, "... N_q d_model, ... N_k d_model -> ... N_q N_k") # shape (B, N_q, N_k)
    D = (O * dO).sum(-1) # shape (B, N_q)
    # dS = P * (dP - D.reshape(B, N_q, 1).expand(B, N_q, N_k)) # shape (B, N_q, N_k)
    dS = P * (dP - D.reshape(B, N_q, 1)) # shape (B, N_q, N_k)
    dQ = einsum(dS, K.transpose(1, 2), "... N_q N_k, ... d_model N_k -> ... N_q d_model") / math.sqrt(d_model) # shape (B, N_q, d_model)
    dK = einsum(dS.transpose(1, 2), Q.transpose(1, 2), "... N_k N_q, ... d_model N_q -> ... N_k d_model") / math.sqrt(d_model) # shape (B, N_k, d_model)
    return dQ, dK, dV


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # will loop over tiled K, V
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # shape B_q, D
    Oi = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)

    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # shape B_k, D
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # shape B_k, D
        offs_k = K_TILE_SIZE * j + tl.arange(0, K_TILE_SIZE)

        causal_mask = offs_k[None, :] <= offs_q[:, None]

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale # shape B_q, B_k
        if is_causal:
            Sij = tl.where(causal_mask, Sij, -float("inf"))

        mi_new = tl.maximum(mi, tl.max(Sij, axis=1))

        P = tl.exp(Sij - mi_new[:, None]) # mij[:, None] or mij.expand_dims(1) shape [B_q, 1], P shape B_q, B_k

        alpha = tl.exp(mi - mi_new) # shape B_q
        li_new = alpha * li + tl.sum(P, axis=1) # shape B_q

        P_casted = P.to(Vj.dtype)
        Oi_new = alpha[:, None] * Oi
        Oi_new = tl.dot(P_casted, Vj, Oi_new) # shape B_q, D

        # Move to next block
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        mi = mi_new
        li = li_new
        Oi = Oi_new

    Oi = (1 / li)[:, None] * Oi
    # Write to O
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))

    Li = mi + tl.log(li)
    # Write to L
    tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0,))


class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, N_q, d_model = Q.shape

        O = torch.zeros((batch, N_q, d_model))
        L = torch.zeros((batch, N_q))

        for b in range(batch):
            Ob, Lb = flash_attention_single_batch(Q[b], K[b], V[b])
            O[b] = Ob
            L[b] = Lb

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO, is_causal=False):
        # Q shape (B, N_q, d_model)
        # K, V shape (B, N_k, d_model)
        # dO shape (B, N_q, d_model)
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = attention_backward(Q, K, V, O, dO, L)
        return dQ, dK, dV, None


class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, N_q, d_model = Q.shape
        _, N_k, _ = K.shape

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal

        O = torch.zeros((batch, N_q, d_model), device="cuda")
        L = torch.zeros((batch, N_q), device="cuda")

        flash_fwd_kernel[(triton.cdiv(N_q, ctx.Q_TILE_SIZE), batch)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            1 / math.sqrt(d_model),
            d_model,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        return O


def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    return FlashAttentionFunc


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    return FlashAttentionTritonFunc


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDP(module)
    raise NotImplementedError


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training, including weight sharding,
    all-gather for forward/backward, and gradient reduce-scatter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with FSDP.
        compute_dtype: optional torch.dtype
            If provided, weights are cast to this dtype before communication
            and compute, saving bandwidth. Master weights stay in fp32.
    Returns:
        Instance of an FSDP class.
    """
    # For example: return FSDP(module, compute_dtype=compute_dtype)
    raise NotImplementedError


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the FSDP-wrapped model.
    """
    # For example: fsdp_model.finish_gradient_synchronization()
    raise NotImplementedError


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors. Replicated parameters are returned as-is.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
    Returns:
        State dictionary mapping parameter names to full (unsharded) tensors.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
