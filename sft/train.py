import argparse
import os

from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
import transformers
import datasets
import optax
import jax
from flax.training.train_state import TrainState

from tqdm.auto import tqdm
import wandb
import datetime
from socket import gethostname

from .data.loss_masked_loader import get_loss_masked_dataloader
from .model.clm import jit_train_step, evaluate_reward_model
from .model.partition_utils import get_sharding_scheme, device_put_leaf

EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 125))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_hf_model", required=True)
    parser.add_argument(
        "--tokenizer",
        required=False,
        default=None,
        help="same as base_hf_model if not specified",
    )
    parser.add_argument("--hf_dataset_dict", required=True)
    parser.add_argument("--max_learning_rate", required=True, type=float)
    parser.add_argument("--train_batch_size", required=True, type=int)
    parser.add_argument("--train_block_size", required=True, type=int)
    parser.add_argument("--train_prng_seed", required=False, type=int, default=0)
    parser.add_argument("--early_stop_threshold", required=False, type=int, default=-1)
    parser.add_argument("--num_epochs", required=False, type=float, default=1.0)
    args = parser.parse_args()

    base_hf_model_name: str = args.base_hf_model
    hf_dataset_dict: str = args.hf_dataset_dict
    max_learning_rate: float = args.max_learning_rate
    train_batch_size: int = args.train_batch_size
    block_size: int = args.train_block_size
    train_prng_seed: int = args.train_prng_seed
    early_stop_threshold: int = args.early_stop_threshold
    num_epochs: float = args.num_epochs

    hf_tokenizer_name: str = args.tokenizer
    if hf_tokenizer_name is None:
        hf_tokenizer_name = base_hf_model_name
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)

    dataset_dict = datasets.load_from_disk(hf_dataset_dict)
    assert isinstance(dataset_dict, datasets.DatasetDict)

    train_ds = dataset_dict["train"]
    eval_ds = dataset_dict["validation"]

    init_train_dataloader, num_train_batches = get_loss_masked_dataloader(
        train_ds, tokenizer, block_size, train_batch_size, num_epochs, train_prng_seed
    )
    init_eval_dataloader, num_eval_batches = get_loss_masked_dataloader(
        eval_ds, tokenizer, block_size, train_batch_size
    )

    model, params_cpu = FlaxAutoModelForCausalLM.from_pretrained(
        base_hf_model_name, _do_init=False
    )  # type: ignore
    assert isinstance(model, (transformers.FlaxPreTrainedModel))

    lr_schedule = optax.linear_schedule(max_learning_rate, 0.0, num_train_batches)
    optimizer = optax.adamw(lr_schedule, eps=1e-6, eps_root=1e-6)

    # Shard and initialize model parameters.
    sharding_scheme = get_sharding_scheme(params_cpu, num_replicas=1)
    initial_params = jax.tree_util.tree_map(
        device_put_leaf, params_cpu, sharding_scheme
    )
    random_state_prng_key = jax.random.PRNGKey(train_prng_seed)
    params = jax.jit(model.init_weights, static_argnames=["input_shape"])(
        random_state_prng_key, (1, 1), initial_params
    )
    opt_state = optimizer.init(params)
    apply_fn = model.__call__  # type: ignore
    train_state = TrainState(0, jax.jit(apply_fn), params, optimizer, opt_state)

    # Initialize wandb
    wandb_run_name = datetime.datetime.now().isoformat() + "-" + gethostname()
    wandb.init(project="instruct_sft", name=wandb_run_name)

    for batch in tqdm(
        init_train_dataloader(), total=num_train_batches, desc="Training", ncols=80
    ):
        stats = {}
        prev_losses = []

        if train_state.step % EVAL_EVERY == 0:
            eval_loss = evaluate_reward_model(
                train_state.apply_fn,
                train_state.params,
                init_eval_dataloader(),
                num_eval_batches,
            )
            stats["validation_loss"] = eval_loss

            if len(prev_losses) > 0 and early_stop_threshold > 0:
                if eval_loss > max(prev_losses[-early_stop_threshold:]):
                    wandb.log(stats)
                    wandb.finish()

        train_state, loss = jit_train_step(train_state, batch)
        stats["train_loss"] = loss.item()

        wandb.log(stats)
