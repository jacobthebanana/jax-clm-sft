"""
Data loaders with a loss mask.
The prompt part of the input is masked for CLM loss.
CLM loss is calculated only over the output.
"""
from typing import NamedTuple, Union, Tuple, Callable, Iterator, List
import datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import jax
import numpy as np

NumBatches = int
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class LossMaskedBatch(NamedTuple):
    input_ids: jax.numpy.ndarray  # int (batch, length)
    attention_mask: jax.numpy.ndarray  # int (batch, length)
    loss_mask: jax.numpy.ndarray  # int (batch, length)


def merge_and_tokenize(
    prompts: List[str],
    responses: List[str],
    block_size: int,
    delimiter: str,
    tokenizer: Tokenizer,
) -> LossMaskedBatch:
    """
    Merge and tokenize prompt and response texts.
    Returns a token-level loss mask, where all tokens are 0 except for
    tokens from the responses.
    """
    delimiter_token_input_id = tokenizer(delimiter).input_ids[-1]

    assert len(prompts) == len(
        responses
    ), "Make sure the number of prompts matches number of responses"
    num_pairs = len(prompts)

    merged_texts: List[str] = []
    for prompt, response in zip(prompts, responses):
        merged_text = delimiter.join([prompt, response])
        merged_texts.append(merged_text + delimiter)

    tokenizer_output = tokenizer(
        merged_texts,
        return_tensors="np",
        max_length=block_size,
        padding="max_length",
        truncation=True,
    )

    input_ids = tokenizer_output.input_ids
    attention_mask = tokenizer_output.attention_mask

    assert len(input_ids.shape) == 2
    assert input_ids.shape[0] == num_pairs

    loss_mask = np.zeros((num_pairs, block_size))
    for pair_index, pair_input_ids in enumerate(input_ids):
        # Set mask to "1" for all tokens between (not including)
        # the two delimiters.
        delimiter_matches = np.argwhere(pair_input_ids == delimiter_token_input_id)

        # First response token (not including delimiter.)
        index_a = delimiter_matches[1].item() + 1
        index_b = delimiter_matches[-1].item()

        loss_mask[pair_index, index_a:index_b] = 1
        attention_mask[pair_index, index_b] = 0

    return LossMaskedBatch(
        jax.numpy.array(input_ids),
        jax.numpy.array(attention_mask),
        jax.numpy.array(loss_mask),
    )


def get_loss_masked_tokenizer(
    dataset: datasets.Dataset,
    tokenizer: Tokenizer,
    block_size: int,
    batch_size: int,
    num_epochs: float = 1.0,
    jax_prng_seed: int = 0,
    delimiter: str = "</s>",
) -> Tuple[Callable[[], Iterator[LossMaskedBatch]], NumBatches]:
    """
    CLM loss is calculated only over the output.

    Returns iterator initializer and number of batches.
    """
    if not isinstance(dataset, datasets.Dataset):
        if isinstance(dataset, datasets.DatasetDict):
            raise AssertionError("Must specify a dataset split")
        raise AssertionError("Must supply an HF arrow dataset.")

    num_examples = len(dataset)
    num_examples_returned = int(len(dataset) * num_epochs)
    num_batches = num_examples_returned // batch_size

    def _initialize_dataloader() -> Iterator[LossMaskedBatch]:
        indices = np.arange(num_examples_returned)
        prng_key = jax.random.PRNGKey(jax_prng_seed)
        shuffled_indices = jax.random.permutation(prng_key, indices)

        for batch_index in range(num_batches):
            index_a = batch_index * batch_size
            index_b = index_a + batch_size

            selected_indices = shuffled_indices[index_a:index_b]
            batch_dataset_indices = list(
                map((lambda x: x % num_examples), selected_indices)
            )
            data_batch = dataset[batch_dataset_indices]
            yield merge_and_tokenize(
                data_batch["prompt"],
                data_batch["response"],
                block_size,
                delimiter,
                tokenizer,
            )

    return _initialize_dataloader, num_batches
