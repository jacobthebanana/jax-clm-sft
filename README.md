# Semi-Supervised Instruction Fine-Tuning with JAX
## Quick Start
### Packages
```bash
python3 -m venv env
source env/bin/activate

pip3 install transformers  # 4.26.1
pip3 install datasets  # 2.9.0
pip3 install flax  # 0.6.4

pip3 install wandb
```

Important: Install jax and jaxlib as described in [JAX documentations](https://jax.readthedocs.io/en/latest/index.html#installation).


### Data Preprocessing
```bash
mkdir -pv data/raw
mkdir -pv data/processed

git lfs install
git clone https://huggingface.co/datasets/Hello-SimpleAI/HC3 data/raw/HC3

python3 -m sft.data.convert_hc3_dataset \
    data/raw/HC3/reddit_eli5.jsonl \
    data/processed/reddit_eli5
```

### Basic Fine-Tuning
```bash
python3 -m sft.train \
    --base_hf_model=jacobthebanana/galactica-125m \
    --early_stop_threshold=5 \
    --hf_dataset_dict=data/processed/eli5 \
    --max_learning_rate=1e-05 \
    --num_epochs=1 \
    --train_batch_size=8 \
    --train_block_size=256 \
    --train_prng_seed=0
```

## Contributing
### Unit Testing
```bash
JAX_DEBUG_NANS=True python3 -m unittest sft.tests
```

## Acknowledgements
> Research supported with compute resources from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/)