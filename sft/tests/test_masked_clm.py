import unittest
import os

import datasets
from transformers import FlaxOPTForCausalLM, AutoTokenizer

import jax
import optax
from flax.training.train_state import TrainState

from ..data.loss_masked_loader import get_loss_masked_tokenizer
from ..model.clm import jit_loss_logit_fn, logit_loss_grad_fn, jit_train_step

HF_MODEL_NAME = os.environ.get("TEST_HF_MODEL", "jacobthebanana/galactica-125m")
DELIMITER = os.environ.get("DELIMITER", "<s>")
HF_TOKENIZER_NAME = os.environ.get("TEST_HF_TOKENIZER", HF_MODEL_NAME)
EXAMPLE_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "8"))
EXAMPLE_BLOCK_SIZE = int(os.environ.get("TEST_BLOCK_SIZE", "512"))
NUM_TRAIN_STEPS = int(os.environ.get("TEST_NUM_TRAIN_STEPS", "12"))
LEARNING_RATE = float(os.environ.get("TEST_LEARNING_RATE", "0.001"))
DATASET_PATH = os.environ.get("TEST_DATASET_DICT_PATH", "data/processed/reddit_eli5")


class MaskedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_dict = datasets.load_from_disk(DATASET_PATH)
        assert isinstance(dataset_dict, datasets.DatasetDict)
        dataset = dataset_dict["test"]
        tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)

        tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
        cls.init_train_dataloader, _ = get_loss_masked_tokenizer(
            dataset,
            tokenizer,
            EXAMPLE_BLOCK_SIZE,
            EXAMPLE_BATCH_SIZE,
            1.7,
            0,
            DELIMITER,
        )

        cls.model, initial_params = FlaxOPTForCausalLM.from_pretrained(
            HF_MODEL_NAME, _do_init=False
        )  # type: ignore
        assert isinstance(cls.model, FlaxOPTForCausalLM)

        initial_params = jax.device_put(initial_params, jax.devices()[0])
        cls.params = jax.jit(cls.model.init_weights, static_argnames=["input_shape"])(
            jax.random.PRNGKey(0),
            (1, 1),
            initial_params,  # type: ignore
        )

    def test_loss_logit_fn(self):
        dataloader = MaskedTrainingTests.init_train_dataloader()
        example_batch = next(dataloader)

        loss, logits = jit_loss_logit_fn(
            MaskedTrainingTests.params,
            MaskedTrainingTests.model.__call__,
            example_batch,
        )

        print(loss, logits)

    def test_gradient_fn(self):
        dataloader = MaskedTrainingTests.init_train_dataloader()
        example_batch = next(dataloader)
        print(example_batch.loss_mask)
        print(jax.numpy.sum(example_batch.loss_mask))

        (loss, logits), gradients = logit_loss_grad_fn(
            MaskedTrainingTests.params,
            MaskedTrainingTests.model.__call__,
            example_batch,
        )

        optimizer = optax.adamw(LEARNING_RATE, eps=1e-6, eps_root=1e-6)
        opt_state = optimizer.init(MaskedTrainingTests.params)
        optimizer.update(gradients, opt_state, MaskedTrainingTests.params)

        print(jax.tree_util.tree_map(jax.numpy.sum, gradients))
        print(loss)

    def test_step_train_state(self):
        dataloader = MaskedTrainingTests.init_train_dataloader()
        example_batch = next(dataloader)

        optimizer = optax.adamw(LEARNING_RATE, eps=1e-6, eps_root=1e-6)
        opt_state = optimizer.init(MaskedTrainingTests.params)
        train_state = TrainState(
            0,
            MaskedTrainingTests.model.__call__,
            MaskedTrainingTests.params,
            optimizer,
            opt_state,
        )

        for _ in range(NUM_TRAIN_STEPS):
            train_state, loss = jit_train_step(train_state, example_batch)
            print(loss.item())
