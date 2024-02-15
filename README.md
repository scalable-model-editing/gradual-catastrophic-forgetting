# Gradual Catastrophic Forgetting

Studying the effects of model editing at scale.

## Installation
We work off of the [MEMIT](https://github.com/kmeng01/memit) codebase, so we'll reference the same installation procedures here: 
"We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`."

## Running the experiments
Two main scripts are used to run the evaluation suite. The first being model-editing with the editing-proficiency and downstream evaluation tasks. This can be done with:
```python
python experiments/evaluate_glue.py \
    --sample_num=0 \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --ds_name=cf \
    --glue_eval_interval=5 \
    --model_save_interval=20 \
    --model_save_location=/path/to/storage \
```
To run evaluations on the history, i.e previous edits made to the model:
```python
python experiments/evaluate_history.py \
    --sample_num=0 \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --ds_name=cf \
    --model_save_interval=20 \
    --continue_from_run=run_n
    --model_save_location=/path/to/storage \
```
Other valid arguments for ```sample_num``` include 0-9. Other algorithms, ```alg_name```, include "ROME", "MEND", and "FT" (Finetuning). The list of compatible models for ```model_name``` include "gpt2-medium", "gpt2-large", "gpt2-xl", and "EleutherAI/gpt-j-6B". For any choice of model, update the ```hparams_fname``` to the json file found in ```hparams/alg_name/```. List of possible datasets for ```ds_name``` include "mcf" (MultiCounterFact), "cf" (CounterFact), "zsre" (zsRE). The ```glue_eval_interval``` specifies the interval of edits made for evaluation on downstream tasks. ```model_save_interval``` is the number of edits made between model saves. The argument for ```model_save_location``` should be the path to the directory where model save should happen. Be sure to include file paths to <ins>unique</ins> directories for each run (eg. ```/data/edited_models/alg_name/run_n```) to avoid conflicts with differing runs. ```continue_from_run``` is a required argument for history evaluations; for a given run, the value is found by ```results/alg_name/run_n``` once experiments have been run using evaluate_glue. Other optional arguments can be found in either files. By default, the experiments run with ```ROME``` on ```gpt2-xl``` with sample ```0```.
