import unittest
import os

import datasets
import jax
import numpy as np
from transformers import AutoTokenizer
import datasets

from ..data.loss_masked_loader import merge_and_tokenize, get_loss_masked_tokenizer

DATASET_DICT_PATH = os.environ.get(
    "TEST_DATASET_DICT_PATH", "data/processed/reddit_eli5"
)
DATASET_DICT_SPLIT = os.environ.get("TEST_DATASET_DICT_SPLIT", "test")
TOKENIZER_PATH = os.environ.get("TEST_TOKENIZER", "jacobthebanana/galactica-6.7b")
EXAMPLE_BATCH_SIZE = int(os.environ.get("TEST_EXAMPLE_BATCH_SIZE", "16"))
EXAMPLE_BLOCK_SIZE = int(os.environ.get("TEST_EXAMPLE_BLOCK_SIZE", "512"))


def _print_tree_shape(tree):
    shape = jax.tree_util.tree_map(jax.numpy.shape, tree)
    print(shape)


class LossMaskedDataloaderTestCases(unittest.TestCase):
    def setUp(self):
        self.test_dataset: datasets.Dataset = datasets.load_from_disk(
            DATASET_DICT_PATH
        )[
            DATASET_DICT_SPLIT
        ]  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def test_merge_and_tokenize(self):
        example_prompts = ["Example Prompt", "Example Prompt A"]
        example_responses = ["Example Response", "Example Response A"]
        example_delimiter = "<s>"

        processed_batch = merge_and_tokenize(
            example_prompts,
            example_responses,
            EXAMPLE_BLOCK_SIZE,
            example_delimiter,
            self.tokenizer,
        )

        selected_tokens = np.array(
            jax.numpy.where(processed_batch.loss_mask, processed_batch.input_ids, 0)
        )

        for pair_index, response in enumerate(example_responses):
            pair_tokens = selected_tokens[pair_index]
            response_tokens = pair_tokens[pair_tokens != 0]
            response_decoded = self.tokenizer.decode(
                response_tokens, skip_special_tokens=True
            )
            assert response == response_decoded

    def test_dataloader(self):
        init_dataloader, num_batches = get_loss_masked_tokenizer(
            self.test_dataset,
            self.tokenizer,
            EXAMPLE_BLOCK_SIZE,
            EXAMPLE_BATCH_SIZE,
            1.7,
        )
        example_batch = next(init_dataloader())

        print(example_batch.loss_mask)
        print(example_batch.loss_mask.sum())
        # print(num_batches)
        # _print_tree_shape(example_batch)
        # print(example_batch.loss_mask.sum(axis=-1))
