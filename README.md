# LMSYS 3rd Place Solution

## Setup

- Trained on combination of 2x4090s (AMD Ryzen 9 7950X 16-Core Processor), 4-8x A100s on vast.ai and 8xH100s on Lambda
- Ubuntu 22.04
- Python environment specs in `environment.yml`

## Process training data



## Training the model

Run `run_stage_1.sh` to train the first stage models and generate pseudo labels. Then, run `run_stage_2.sh` to train the pseudo labeled models.

## Running inference on a new dataset

One way is to replace the input dataset in our submission notebook. Another way is, in `llm_validate.py`, replace `train.parquet` (which is the processed version of the competition dataset) with your own dataset and run `llm_validate.py`. If you want TTA'd predictions, run it with the `--tta` flag and ensemble the saved files.