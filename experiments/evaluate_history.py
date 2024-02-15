import json
import shutil
from itertools import islice
import time
from typing import Tuple, Union
import sys
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('/path/to/gradual-catastrophic-forgetting')
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from dsets.counterfact import CounterFactDataset
from util import nethook
from util.globals import *

from glue_eval.glue_eval import GLUEEval

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    model_save_location: str,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory, guarantee models from previous run are stored
    assert os.path.exists(model_save_location), "Invalid model save location"
    print(f"Getting models from {model_save_location}")
    assert (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists(), "Invalid run_id, no run was recorded with that run_id"
    print(f"Storing history evaluations in {str(run_dir)}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    #load indices file and initialize dataset class
    f = open(args.indices_filename)
    sampled_indices = json.load(f)
    dataset = CounterFactDataset('data')

    # Iterate through dataset
    
    edits_so_far = []
    fact_remembered_flag = {}
    previous_history_folder = None
    for e, element_index in enumerate(sampled_indices[args.sample_num]):
        if e >= 1200: break # added for testing 
        datapoint = dataset.__getitem__(element_index)
        record_chunks = [datapoint]
        edits_so_far.append(datapoint)
        case_result_template = str(run_dir / "case_{}.json")
        case_ids = [record["case_id"] for record in record_chunks]

        # retrieve the next model that has seen the current data, used last stored model if we get to the end
        model_save_folder = model_save_location + '/edits_' + str(e + args.model_save_interval)
        if (e) % args.model_save_interval == 0 and os.path.exists(model_save_folder):  
            print(f"Updating model to model stored at {model_save_folder}")               
            if conserve_memory:
                edited_model = AutoModelForCausalLM.from_pretrained(model_save_folder, device_map='cpu')
            else:
                edited_model = AutoModelForCausalLM.from_pretrained(model_save_folder).cuda()


        ##### Evaluate previous edits, only do it on saved model intervals
        if (e + 1) % args.model_save_interval == 0:
            print(f"Evaluating history for all previous {str(e+1)} edits")
            start = time.time()
            gen_test_vars = [snips, vec]

            history_save_location = str(run_dir) + '/' + 'history_eval_' + str(e + 1) + '/'
            print(history_save_location)
            os.makedirs(history_save_location, exist_ok=True)
            
            actual_edits = 0
            for r, history_record in enumerate(edits_so_far):
                out_file = Path(history_save_location + str(r) + '_' + "case_{}.json".format(history_record["case_id"]))
                if out_file.exists():

                    #save if fact previously forgotten or not
                    with open(out_file, "r") as f:
                        data = json.load(f)
                    fact_remembered_flag[history_record["case_id"]] = data['post']['rewrite_prompts_correct'][0]
                    
                    print(f"Skipping {out_file}; already exists")
                    continue

                if history_record["case_id"] in fact_remembered_flag and not fact_remembered_flag[history_record["case_id"]]:
                    #if fact forgotten previously, copy from previous history
                    source_path = Path(previous_history_folder + str(r) + '_' + "case_{}.json".format(history_record["case_id"]))
                    destination_path = out_file
                    shutil.copyfile(source_path, destination_path)
                    print('copying...', source_path, destination_path)
                    continue

                actual_edits += 1
                print(actual_edits, r)
                metrics = {
                    "case_id": history_record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": history_record["requested_rewrite"],
                    "time": None,
                    "post": ds_eval_method(
                        edited_model,
                        tok,
                        history_record,
                        *(
                            gen_test_vars
                            if history_record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases, don't have time since not making edit in this iteration
                    ),
                }

                #save if fact remembered
                fact_remembered_flag[history_record["case_id"]] = metrics['post']['rewrite_prompts_correct'][0]


                # Dump metrics in .json
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)

            previous_history_folder =  history_save_location      
            print("Evaluation took", time.time() - start)
            print('ACTUAL EDITS:', actual_edits)

            edited_model.cpu()
        




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_num",
        type=str,
        default="0",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--indices_filename",
        type=str,
        default="counterfact_sampled_unique_10_20391.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--model_save_interval",
        type=int,
        default=20,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
        required=True
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--model_save_location",
        type=str,
        default=None,
        help="Location of saved models for a given run",
        required=True,
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        model_save_location=args.model_save_location,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )
