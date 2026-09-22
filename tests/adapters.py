from __future__ import annotations

import math
import torch

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
            P = (Sij - mi_new.reshape((B_q, 1)).expand(B_q, B_k)).exp() # shape (B_q, B_k)

            li_new = (mi - mi_new).exp() * li + P.sum(dim=-1) # shape (B_q)

            Oi = (mi - mi_new).exp().diag() @ Oi + P @ Vj # shape (B_q, d_model)

            mi = mi_new
            li = li_new

        O[i*B_q:(i+1)*B_q] = (1 / li).diag() @ Oi
        L[i*B_q:(i+1)*B_q] = mi + li.log()

    return O, L


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

        ctx.save_for_backward(L)
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
    raise NotImplementedError


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
