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
**Before any experiment is run**, be sure to update ```sys.path.append('/path/to/gradual-catastrophic-forgetting')``` to the path of the parent directory. 

By default, either experiment performs 1200 edits. To make a change to this, navigate to the ```if e >= 1200: break```. Other valid arguments:
- ```sample_num```:  0-9
- ```alg_name```: "ROME", "MEMIT", "MEND", and "FT" (Finetuning)
- ```model_name```: "gpt2-medium", "gpt2-large", "gpt2-xl", and "EleutherAI/gpt-j-6B". For any choice of model, update the ```hparams_fname``` to the json file found in ```hparams/alg_name/```
- ```ds_name```: "mcf" (MultiCounterFact), "cf" (CounterFact), "zsre" (zsRE)
- ```glue_eval_interval```: interval of edits made between each evaluation on downstream tasks
- ```model_save_interval```: interval of edits made between each model save
- ```model_save_location```: path to the directory where model save should happen. Be sure to include file paths to <ins>unique</ins> directories for each run (eg. ```/data/edited_models/alg_name/run_n```) to avoid conflicts with differing runs
- ```continue_from_run```: required argument for history evaluations. For a given run, the value is found by ```results/alg_name/run_n``` once experiments have been run using evaluate_glue.

Other optional arguments can be found in either files. By default, the experiments run with ```ROME``` on ```gpt2-xl``` with sample ```0```.

To continue onward from a checkpoint or if a run is terminated before completion, use the ```experiments/evaluate_glue_restart.py``` file. To choose a checkpoint, set the ```restart_index``` to the desired model. For example, the model stored in ```/edits_100``` would have a ```restart_index``` of ```100```. If this is not provided, the script will find the last model. Again, ```continue_from_run``` is required to store the future results.

To visualize the plots in the paper, navigate to ```downstream_eval/current_edit_scores.py```,```downstream_eval/glue_performance.py```, and ```downstream_eval/fact_forgetting.py```. Update the ```algo```, ```run```, and ```sample_num``` to the desired run to generate the plots. If the experiment performs a different number of edits, be sure to change the ```xlim``` from 1200.
 
