# LMSYS 3rd Place Solution

## Setup

- Trained on combination of 2x4090s (AMD Ryzen 9 7950X 16-Core Processor), 4-8x A100s/4090s/H100s on vast.ai and 8xH100s on Lambda
- Ubuntu 22.04
- Python (conda) environment specs in `environment.yml`

## Process training data

Download the competition data into the `data` directory. It's expecting a file called `train.csv` and `test.csv`. Also, create a `preds` and `pseudo` directory under `data`. Then, run `process_data.py`, which will download the additional datasets used for training. You may need to locally authenticate with Hugging Face Hub to do this, using `huggingface-cli login`.

## Getting lmsys-1m paired completions for pseudo labeling

Run `scripts/vllm_generate.py` to generate paired completions for the lmsys-1m dataset.

## Training the model

Run `run_stage_1.sh` to train the first stage models and generate pseudo labels. Then, run `run_stage_2.sh` to train the pseudo labeled models. Note that the second stage (pseudo label) configs assume you are running on 8 GPUs, whereas the first stage models assume you are running on 2 GPUs. If you want to adjust the number of GPUs, adjust the batch size and accum parameters in the configs accordingly so that they are running on an effective batch size of 8. Specifically, these settings:

```
training:
  batch_size: 4
  accum: 2
  max_length: 1800
```

## Running inference on a new dataset

One way is to replace the input dataset in our submission notebook. Another way is, in `llm_validate.py`, replace `train.parquet` (which is the processed version of the competition dataset) with your own dataset and run `llm_validate.py`. If you want TTA'd predictions, run it with the `--tta` flag and ensemble the saved files.