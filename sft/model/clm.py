"""
Training loop for Causal Language Modelling. 
"""
from typing import Any, Tuple, Callable, NamedTuple, Iterator, Optional, List
from transformers import FlaxAutoModelForCausalLM
from flax.training.common_utils import onehot
from flax.training.train_state import TrainState
import optax

import jax
import jax.numpy as jnp

from tqdm.auto import tqdm

from ..data.loss_masked_loader import LossMaskedBatch

Params = Any
Gradients = Any
Loss = jax.numpy.ndarray
Logits = jax.numpy.ndarray
LossFloat = float


def loss_logit_fn(
    params: Any, apply_fn: Callable, batch: LossMaskedBatch
) -> Tuple[Loss, Logits]:
    logits = apply_fn(
        batch.input_ids, attention_mask=batch.attention_mask, params=params
    ).logits

    # Logic for per-token loss reproduced from huggingface/transformers
    shift_logits = logits[..., :-1, :]  # (batch, tokens - 1, vocab)
    shift_labels = batch.input_ids[..., 1:]  # (batch, tokens - 1)
    shift_loss_mask = batch.loss_mask[..., 1:]  # (batch, tokens - 1)

    loss = (
        optax.softmax_cross_entropy(
            shift_logits, onehot(shift_labels, shift_logits.shape[-1])
        )
        * shift_loss_mask
    ).sum()

    num_tokens_selected = jnp.sum(shift_loss_mask)

    # Return loss averaged over num_tokens_selected if num_tokens_selected > 0.
    # Return 0. otherwise.
    return (
        jnp.where(num_tokens_selected >= 1, loss / (num_tokens_selected + 1e-5), 0.0),
        logits,
    )


def jit_loss_logit_fn(
    params: Any, apply_fn: Callable, batch: LossMaskedBatch
) -> Tuple[Loss, Logits]:
    """
    Calculate loss only over logits selected in batch.loss_mask.

    Returns loss and predicted logits.
    """
    ...


jit_loss_logit_fn = jax.jit(loss_logit_fn, static_argnames=["apply_fn"])


def logit_loss_grad_fn(
    params: Any, apply_fn: Callable, batch: LossMaskedBatch
) -> Tuple[Tuple[Loss, Logits], Gradients]:
    """
    Calculate loss only over logits selected in batch.loss_mask.

    Returns loss and predicted logits.
    """
    ...


logit_loss_grad_fn = jax.value_and_grad(loss_logit_fn, argnums=0, has_aux=True)


def train_step(state: TrainState, batch: LossMaskedBatch) -> Tuple[TrainState, Loss]:
    (loss, _), gradients = logit_loss_grad_fn(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=gradients)

    return state, loss


def jit_train_step(
    state: TrainState, batch: LossMaskedBatch
) -> Tuple[TrainState, Loss]:
    ...


jit_train_step = jax.jit(train_step)


def evaluate_reward_model(
    apply_fn: Callable,
    params: Params,
    eval_dataloader: Iterator[LossMaskedBatch],
    num_eval_batches: Optional[int] = None,
) -> LossFloat:
    """
    Evaluate model on the given eval dataloader.
    """
    loss_tally = 0.0
    num_batches = 0

    for batch in tqdm(eval_dataloader, total=num_eval_batches, ncols=80, leave=False):
        num_batches += 1
        loss, _ = jit_loss_logit_fn(params, apply_fn, batch)
        loss_tally += loss.item()

    return loss_tally / num_batches
