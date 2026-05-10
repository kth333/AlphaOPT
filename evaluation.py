import os
import json
import time
import yaml
import statistics
from typing import List, Tuple, Optional, Any
from dotenv import load_dotenv
load_dotenv()

from tqdm.auto import tqdm

import concurrent.futures

from src.utils import cal_time_cost, get_token_usage
from src.dataloader import DataLoader, Task          
from src.llm_programmer import ProgramGenerator
from src.experience_library import ExperienceLibrary
from src.llm_retriever import LibraryRetrieval
from src.train_eval_utils import check_optimality, self_debug 

def evaluate(
    tasks: List["Task"],
    llm_opt: "ProgramGenerator",
    use_library: bool,
    library: Optional["ExperienceLibrary"],
    config: Any,
) -> Tuple[int, int, int, float, dict]:
    """
    Evaluate the task success rate of a learned experience library on a test dataset
    If use_library is False, the library is not used in the evaluation
    
    Returns:
        n_success: number of successful tasks (pass@1)
        n_runnable: number of runnable tasks
        n_total: total number of tasks
        pass_at_k_rate: pass@k success rate
        token_usage_delta: token usage delta for this evaluation
    """
    n_success = 0
    n_runnable = 0
    # dataset = config.dataset
    output_folder = config.output_folder
    pass_at_k = config.pass_at_k # Default to 1 if not specified

    # Track token usage before evaluation
    usage_before = get_token_usage()

    llm_retri = None

    if use_library:
        llm_retri = LibraryRetrieval(
            lib=library,
            model=llm_opt.model,
            service=config.service,
            temperature=llm_opt.temp,
        )

    def process_task(task, output_dirs):
        """
        Process a single task with multiple attempts for pass@k evaluation
        """
        output_path = f"{output_dirs}/nolib"
        retrieved_ins_ids = []
        formulation_ins, program_ins = [], []

        if use_library:
            # Retrieve relevant insights from an archived experience library
            output_path = f"{output_dirs}/lib"
            formulation_ins = llm_retri.retrieve_applicable_insights(
                    task=task,
                    stage="Formulation",
                    config=config,
                    verbose=False,
                    save_data=True,
                    output_path=output_path
                    )
            retrieved_ins_ids = [ins["insight_id"] for ins in formulation_ins if 'insight_id' in ins]

        # Try multiple times for pass@k evaluation
        attempts_results = []
        for attempt in range(pass_at_k):
            attempt_output_path = f"{output_path}_attempt_{attempt + 1}" if pass_at_k > 1 else output_path
            
            candidate_model = llm_opt.generate_formulation(
                    task=task,
                    retrieved_insights=formulation_ins,
                    # rewrite=bool(config.ablation.rewrite),
                    abl_params=config.ablation,
                    verbose=False,
                    save_data=True,
                    output_path=attempt_output_path
                )
            
            if use_library and config.ablation.include_program_insight:
                program_ins = llm_retri.retrieve_applicable_insights(
                        task=task,
                        stage="Program",
                        config=config,
                        formulation=candidate_model,
                        verbose=False,
                        save_data=True,
                        output_path=attempt_output_path
                    )
                
                retrieved_ins_ids.extend([ins["insight_id"] for ins in program_ins if "insight_id" in ins])

            candidate_program, output, runnable, is_time_out = llm_opt.generate_program(
                    task=task,
                    retrieved_insights=program_ins,
                    formulation=candidate_model,
                    abl_params=config.ablation,
                    verbose=False,
                    save_data=True,
                    output_path=attempt_output_path
                )

            # Check optimality
            is_optimal, status, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
            
            # Self-Debug
            if config.ablation.max_debug_retry:
                if status == "run_error":
                    is_optimal, runnable = self_debug(task, candidate_program, feedback, config)

            attempts_results.append((int(is_optimal), int(runnable), status))
            
            # If we found a successful solution, we can stop early for pass@k
            if is_optimal:
                break

        # Record task (use the first attempt's results for recording)
        task.retri_ins_lst.append(retrieved_ins_ids)
        task.output_status.append(attempts_results[0][2])  # Use first attempt's status

        # Calculate pass@k results
        pass_at_k_success = any(result[0] for result in attempts_results)  # Any attempt succeeded
        pass_at_k_runnable = any(result[1] for result in attempts_results)  # Any attempt was runnable
        
        return int(attempts_results[0][0]), int(attempts_results[0][1]), int(pass_at_k_success), int(pass_at_k_runnable)
    
    output_dirs = [f"testing/{output_folder}/task_{task.id}" for task in tasks]
    # Use ThreadPoolExecutor to process tasks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = list(tqdm(executor.map(process_task, tasks, output_dirs), total=len(tasks), desc="Evaluating\n"))

    # Calculate the number of successes and successful executions from the results
    n_success = sum(opt for opt, _, _, _ in results)  # pass@1 success
    n_runnable = sum(run for _, run, _, _ in results)  # pass@1 runnable
    n_pass_at_k_success = sum(pass_k for _, _, pass_k, _ in results)  # pass@k success
    n_pass_at_k_runnable = sum(pass_k_run for _, _, _, pass_k_run in results)  # pass@k runnable
    
    pass_at_k_rate = n_pass_at_k_success / len(tasks) if len(tasks) > 0 else 0.0
    
    # Calculate token usage delta
    usage_after = get_token_usage()
    token_usage_delta = {}
    for vendor, stats_after in usage_after.items():
        stats_before = usage_before.get(vendor, {})
        token_usage_delta[vendor] = {
            k: float(stats_after.get(k, 0.0) - stats_before.get(k, 0.0))
            for k in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost")
        }
    
    return n_success, n_runnable, len(tasks), pass_at_k_rate, token_usage_delta


def load_config(config_file: str = "eval_config.yaml") -> dict:
    """
    Load configuration from a YAML file
    """
    # with open(config_file, "r") as f:
    #     config = yaml.safe_load(f)  
    #* Configure
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_file)

    #* Generate a timestamp and append it to output_folder
    # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # config.output_folder = f"{config.output_folder}_{ts}"
    # Re-resolve
    # OmegaConf.resolve(config)
    return config


def prepare_dataset_config(base_config: Any, dataset: str) -> Any:
    """
    Prepare configuration for a specific dataset by resolving template variables
    
    Args:
        base_config: Base configuration object
        dataset: Dataset name (must be a string)
        
    Returns:
        Dataset-specific configuration
    """
    from omegaconf import OmegaConf
    
    # Ensure dataset is a string
    dataset = str(dataset).strip()
    
    # Create a copy of the config (convert to dict first to avoid OmegaConf issues)
    config_dict = OmegaConf.to_container(base_config, resolve=False)
    
    # Remove 'datasets' field if it exists to avoid confusion
    if 'datasets' in config_dict:
        del config_dict['datasets']
    
    # Create new config from dict
    dataset_config = OmegaConf.create(config_dict)
    
    # Set dataset-specific values BEFORE resolving
    dataset_config.dataset = dataset
    
    # Resolve data_path and output_folder using the dataset name
    # Check for template fields first, then fall back to regular fields
    if 'data_path_template' in dataset_config:
        template = str(dataset_config.data_path_template)
        dataset_config.data_path = template.replace('${dataset}', dataset)
    elif 'data_path' in dataset_config:
        template = str(dataset_config.data_path)
        dataset_config.data_path = template.replace('${dataset}', dataset)
    else:
        dataset_config.data_path = f"./data/optimization_tasks/clean/{dataset}.json"
    
    if 'output_folder_template' in dataset_config:
        template = str(dataset_config.output_folder_template)
        dataset_config.output_folder = template.replace('${dataset}', dataset)
    elif 'output_folder' in dataset_config:
        template = str(dataset_config.output_folder)
        dataset_config.output_folder = template.replace('${dataset}', dataset)
    else:
        dataset_config.output_folder = f"{dataset}_new_flash"
    
    # Now resolve all variables (dataset is already set as a string)
    try:
        OmegaConf.resolve(dataset_config)
    except Exception as e:
        # If resolution fails, the manual replacements above should still work
        print(f"Warning: OmegaConf.resolve() failed: {e}, using manual replacements")
    
    return dataset_config


def evaluate_single_dataset(config: Any, dataset: str) -> dict:
    """
    Evaluate a single dataset
    
    Args:
        config: Base configuration
        dataset: Dataset name to evaluate
    """
    # Prepare dataset-specific configuration
    dataset_config = prepare_dataset_config(config, dataset)

    # Optional ablation tag (for output/log disambiguation)
    ablation_tag = str(getattr(dataset_config, "ablation_tag", "") or "").strip()
    if ablation_tag:
        # Make sure different ablation runs do not overwrite the same output folder
        dataset_config.output_folder = f"{dataset_config.output_folder}__{ablation_tag}"
    
    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {dataset}")
    print(f"Data path: {dataset_config.data_path}")
    print(f"Output folder: {dataset_config.output_folder}")
    if ablation_tag:
        print(f"Ablation tag: {ablation_tag}")
    print(f"{'='*60}\n")

    # How many times to evaluate the same dataset (for mean/min/max)
    try:
        n_runs = int(getattr(dataset_config, "n_runs", 1) or 1)
    except Exception:
        n_runs = 1
    if n_runs < 1:
        n_runs = 1

    # Check if library_path is provided; if not, set use_library flag to False
    use_library = bool(dataset_config.library_path)

    if use_library:
        # Load trained experience library
        print("Loading Library...")
        library = ExperienceLibrary.from_json_file(dataset_config.library_path)
    else:
        print("Do task without Library...")
        library = None

    # Initialize ProgramGenerator
    llm_opt = ProgramGenerator(
        model       = dataset_config.model,
        service     = dataset_config.service,
        temperature = dataset_config.temperature,
    )

    def _summarize(vals: List[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(statistics.mean(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    base_output_folder = str(dataset_config.output_folder)
    pass_at_k = dataset_config.pass_at_k

    per_run_results: List[dict] = []

    for run_idx in range(1, n_runs + 1):
        run_output_folder = f"{base_output_folder}/run_{run_idx}" if n_runs > 1 else base_output_folder
        dataset_config.output_folder = run_output_folder

        # Load test tasks fresh each run (avoid accumulating records in memory)
        test_tasks = DataLoader(dataset_config.data_path, mode="test")

        test_tasks_save_path = (
            f"./testing/{run_output_folder}/tasks_record_lib.json"
            if use_library
            else f"./testing/{run_output_folder}/tasks_record_nolib.json"
        )

        print(f"\n--- Run {run_idx}/{n_runs} ---")
        print(f"Output folder (run): {run_output_folder}")

        # Run evaluation
        start_time = time.time()
        n_success, n_runnable, n_total, pass_at_k_rate, token_usage_delta = evaluate(
            test_tasks, llm_opt, use_library, library, dataset_config
        )
        success_rate = round(n_success / n_total, 3) if n_total else 0.0
        execution_rate = round(n_runnable / n_total, 3) if n_total else 0.0

        # Extract token cost (sum of all non-zero costs from all vendors)
        token_cost = sum(
            stats.get("cost", 0.0)
            for vendor, stats in token_usage_delta.items()
            if stats.get("cost", 0.0) != 0.0
        )

        # Extract token counts (sum across vendors)
        prompt_tokens = sum(float(stats.get("prompt_tokens", 0.0) or 0.0) for stats in token_usage_delta.values())
        completion_tokens = sum(float(stats.get("completion_tokens", 0.0) or 0.0) for stats in token_usage_delta.values())
        total_tokens = sum(float(stats.get("total_tokens", 0.0) or 0.0) for stats in token_usage_delta.values())

        # Count time cost (minutes as float)
        eval_duration = cal_time_cost(start_time, f"Evaluation for {dataset} (run {run_idx}/{n_runs})")

        print(
            f"\n================  EVALUATION RESULT ({dataset}) [Run {run_idx}/{n_runs}]  ================\n"
            f"Tasks evaluated : {n_total}\n"
            f"Pass@1 Success  : {n_success}\n"
            f"Pass@1 Rate     : {success_rate:.3%}\n"
            f"Pass@{pass_at_k} Rate     : {pass_at_k_rate:.3%}\n"
            f"Execution-rate  : {execution_rate:.3%}\n"
            f"Time cost (min) : {eval_duration}\n"
            f"Token cost      : ${token_cost:.6f}\n"
            f"====================================================\n"
        )

        # Save tasks with status record for this run
        test_tasks.save_as_json(test_tasks_save_path)

        # Store run metrics
        per_run_results.append(
            {
                "run_idx": run_idx,
                "output_folder": run_output_folder,
                "n_total": int(n_total),
                "n_success": int(n_success),
                "n_runnable": int(n_runnable),
                "pass_at_1_rate": float(success_rate),
                "pass_at_k_rate": float(pass_at_k_rate),
                "execution_rate": float(execution_rate),
                # Keep both keys for backward compatibility with older logs/scripts
                "duration_min": float(eval_duration),
                "duration": float(eval_duration),
                "token_cost": float(round(token_cost, 6)),
                "prompt_tokens": float(prompt_tokens),
                "completion_tokens": float(completion_tokens),
                "total_tokens": float(total_tokens),
                "token_usage_delta": token_usage_delta,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            }
        )

    # Load existing logs if they exist, otherwise create a new list
    results_path = "./testing/all_test_results.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            if not isinstance(all_results, list):
                print(f"Warning: {results_path} is not a list JSON. Resetting it.")
                all_results = []
        except Exception as e:
            print(f"Warning: Failed to read {results_path}: {e}. Resetting it.")
            all_results = []
    else:
        all_results = []

    # Append per-run results
    for r in per_run_results:
        all_results.append(
            {
                "dataset": dataset_config.dataset,
                "data_path": dataset_config.data_path,
                "library_path": dataset_config.library_path if use_library else "None",
                "model": dataset_config.model,
                "service": dataset_config.service,
                "temperature": dataset_config.temperature,
                "pass_at_k": pass_at_k,
                "n_runs": n_runs,
                "ablation_tag": ablation_tag or "base",
                "run_idx": r["run_idx"],
                "output_folder": r["output_folder"],
                "n_total": r["n_total"],
                "n_success": r["n_success"],
                "n_runnable": r["n_runnable"],
                "pass_at_1_rate": r["pass_at_1_rate"],
                "pass_at_k_rate": r["pass_at_k_rate"],
                "execution_rate": r["execution_rate"],
                "taxonomy": dataset_config.ablation.taxonomy,
                "rewrite": dataset_config.ablation.rewrite,
                "include_example": dataset_config.ablation.include_example,
                "include_program_insight": dataset_config.ablation.include_program_insight,
                "max_debug_retry": dataset_config.ablation.max_debug_retry,
                # Keep both keys for compatibility with existing `all_test_results.json`
                "duration_min": r["duration_min"],
                "duration": r.get("duration", r["duration_min"]),
                "token_cost": r["token_cost"],
                "prompt_tokens": r.get("prompt_tokens", 0.0),
                "completion_tokens": r.get("completion_tokens", 0.0),
                "total_tokens": r.get("total_tokens", 0.0),
                "timestamp": r["timestamp"],
            }
        )

    # Aggregate mean/min/max across runs
    agg = {
        "n_success": _summarize([float(r["n_success"]) for r in per_run_results]),
        "n_runnable": _summarize([float(r["n_runnable"]) for r in per_run_results]),
        "pass_at_1_rate": _summarize([float(r["pass_at_1_rate"]) for r in per_run_results]),
        "pass_at_k_rate": _summarize([float(r["pass_at_k_rate"]) for r in per_run_results]),
        "execution_rate": _summarize([float(r["execution_rate"]) for r in per_run_results]),
        "duration_min": _summarize([float(r["duration_min"]) for r in per_run_results]),
        "token_cost": _summarize([float(r["token_cost"]) for r in per_run_results]),
        "prompt_tokens": _summarize([float(r.get("prompt_tokens", 0.0)) for r in per_run_results]),
        "completion_tokens": _summarize([float(r.get("completion_tokens", 0.0)) for r in per_run_results]),
        "total_tokens": _summarize([float(r.get("total_tokens", 0.0)) for r in per_run_results]),
    }

    print(f"\n================  AGGREGATED RESULT ({dataset}) over {n_runs} run(s)  ================\n"
          f"Pass@1 Rate     : mean={agg['pass_at_1_rate']['mean']:.3%}, min={agg['pass_at_1_rate']['min']:.3%}, max={agg['pass_at_1_rate']['max']:.3%}\n"
          f"Pass@{pass_at_k} Rate     : mean={agg['pass_at_k_rate']['mean']:.3%}, min={agg['pass_at_k_rate']['min']:.3%}, max={agg['pass_at_k_rate']['max']:.3%}\n"
          f"Execution-rate  : mean={agg['execution_rate']['mean']:.3%}, min={agg['execution_rate']['min']:.3%}, max={agg['execution_rate']['max']:.3%}\n"
          f"Time cost (min) : mean={agg['duration_min']['mean']:.3f}, min={agg['duration_min']['min']:.3f}, max={agg['duration_min']['max']:.3f}\n"
          f"Token cost      : mean=${agg['token_cost']['mean']:.6f}, min=${agg['token_cost']['min']:.6f}, max=${agg['token_cost']['max']:.6f}\n"
          f"====================================================\n")

    # Append aggregate entry
    all_results.append(
        {
            "dataset": dataset_config.dataset,
            "data_path": dataset_config.data_path,
            "library_path": dataset_config.library_path if use_library else "None",
            "model": dataset_config.model,
            "service": dataset_config.service,
            "temperature": dataset_config.temperature,
            "pass_at_k": pass_at_k,
            "n_runs": n_runs,
            "aggregate": True,
            "ablation_tag": ablation_tag or "base",
            "taxonomy": dataset_config.ablation.taxonomy,
            "rewrite": dataset_config.ablation.rewrite,
            "include_example": dataset_config.ablation.include_example,
            "include_program_insight": dataset_config.ablation.include_program_insight,
            "max_debug_retry": dataset_config.ablation.max_debug_retry,
            "summary": agg,
            "base_output_folder": base_output_folder,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
        }
    )

    # Save the updated log
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    # Restore output folder to base (avoid surprising callers)
    dataset_config.output_folder = base_output_folder

    return {
        "dataset": dataset_config.dataset,
        "data_path": dataset_config.data_path,
        "library_path": dataset_config.library_path if use_library else "None",
        "model": dataset_config.model,
        "service": dataset_config.service,
        "temperature": dataset_config.temperature,
        "pass_at_k": pass_at_k,
        "n_runs": n_runs,
        "ablation_tag": ablation_tag or "base",
        "base_output_folder": base_output_folder,
        "summary": agg,
    }


def main() -> None:
    # Read the configuration file
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "eval_config.yaml"
    config = load_config(config_file)

    # Get datasets list - support both single string and list
    # Check for 'datasets' first, then fall back to 'dataset' for backward compatibility
    if 'datasets' in config:
        datasets_raw = config.datasets
    elif 'dataset' in config:
        # Backward compatibility: if 'dataset' exists, convert to list
        datasets_raw = [config.dataset]
    else:
        raise ValueError("Configuration must contain either 'datasets' or 'dataset' field")
    
    # Convert to list and ensure all elements are strings
    # Handle OmegaConf ListConfig or regular list
    from omegaconf import ListConfig
    if isinstance(datasets_raw, (list, ListConfig)):
        datasets = [str(d) for d in datasets_raw]
    elif isinstance(datasets_raw, str):
        # If it's a string representation of a list, try to parse it
        if datasets_raw.startswith('[') and datasets_raw.endswith(']'):
            # This shouldn't happen with OmegaConf, but handle it just in case
            import ast
            try:
                datasets = [str(d) for d in ast.literal_eval(datasets_raw)]
            except:
                datasets = [datasets_raw]
        else:
            datasets = [datasets_raw]
    else:
        datasets = [str(datasets_raw)]

    print(f"\n{'='*60}")
    print(f"Starting batch evaluation for {len(datasets)} dataset(s)")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"{'='*60}\n")

    def _summarize(vals: List[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(statistics.mean(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    # Build ablation variants
    from omegaconf import OmegaConf
    from omegaconf import ListConfig
    base_cfg_container = OmegaConf.to_container(config, resolve=False)

    sweep_enabled = bool(getattr(config.ablation, "sweep_one_by_one", False))
    include_base = bool(getattr(config.ablation, "include_base", True))
    sweep_params = getattr(config.ablation, "sweep_params", None)
    if sweep_params is None:
        sweep_params = ["taxonomy", "include_example", "include_program_insight"]
    # OmegaConf uses ListConfig for YAML lists; treat it like a list here.
    sweep_params = [
        str(p)
        for p in (
            list(sweep_params)
            if isinstance(sweep_params, (list, tuple, ListConfig))
            else [sweep_params]
        )
    ]

    variants: List[tuple[str, dict]] = []
    if (not sweep_enabled) or include_base:
        variants.append(("base", {}))
    if sweep_enabled:
        for p in sweep_params:
            variants.append((f"{p}=false", {p: False}))

    # Run: variants × datasets
    all_variant_overalls: List[dict] = []

    def _append_to_results_log(entry: dict) -> None:
        """
        Best-effort append to ./testing/all_test_results.json
        """
        results_path = "./testing/all_test_results.json"
        try:
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = []
            else:
                all_results = []
        except Exception:
            all_results = []

        all_results.append(entry)
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write log to {results_path}: {e}")

    for v_i, (variant_name, overrides) in enumerate(variants, 1):
        print(f"\n{'#'*70}")
        print(f"Starting ablation variant [{v_i}/{len(variants)}]: {variant_name}")
        if overrides:
            print(f"Overrides: {overrides}")
        print(f"{'#'*70}\n")

        # Create variant config (deep-ish copy via container)
        v_cfg = OmegaConf.create(base_cfg_container)
        # Apply overrides
        for k, v in overrides.items():
            if hasattr(v_cfg.ablation, k):
                setattr(v_cfg.ablation, k, v)
            else:
                v_cfg.ablation[k] = v
        v_cfg.ablation_tag = variant_name

        all_dataset_aggs: List[dict] = []

        # Evaluate each dataset under this variant
        for i, dataset in enumerate(datasets, 1):
            dataset = str(dataset).strip()
            if dataset.startswith("[") and dataset.endswith("]"):
                print(f"\n⚠️  Warning: Dataset appears to be a list representation: {dataset}")
                print("   This suggests a configuration parsing issue. Skipping...")
                continue

            print(f"\n[{i}/{len(datasets)}] Processing dataset: {dataset} (variant: {variant_name})")
            try:
                dataset_agg = evaluate_single_dataset(v_cfg, dataset)
                if isinstance(dataset_agg, dict):
                    all_dataset_aggs.append(dataset_agg)
            except Exception as e:
                print(f"\n❌ Error evaluating dataset '{dataset}' (variant: {variant_name}): {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with next dataset...\n")
                continue

        print(f"\n{'='*60}")
        print(f"Variant completed: {variant_name} ({len(all_dataset_aggs)}/{len(datasets)} dataset(s) succeeded)")
        print(f"{'='*60}\n")

        # Per-variant overall summary across datasets (mean/min/max across dataset-level means)
        if all_dataset_aggs:
            def _get_mean_rate(key: str) -> List[float]:
                out: List[float] = []
                for d in all_dataset_aggs:
                    summary = d.get("summary") or {}
                    v = (summary.get(key) or {}).get("mean", None)
                    if v is not None:
                        out.append(float(v))
                return out

            overall = {
                "pass_at_1_rate": _summarize(_get_mean_rate("pass_at_1_rate")),
                "pass_at_k_rate": _summarize(_get_mean_rate("pass_at_k_rate")),
                "execution_rate": _summarize(_get_mean_rate("execution_rate")),
                "duration_min": _summarize(_get_mean_rate("duration_min")),
                "token_cost": _summarize(_get_mean_rate("token_cost")),
            }

            print(
                f"\n================  OVERALL AGGREGATED RESULT ({variant_name}) across {len(all_dataset_aggs)} dataset(s)  ================\n"
                f"Pass@1 Rate     : mean={overall['pass_at_1_rate']['mean']:.3%}, min={overall['pass_at_1_rate']['min']:.3%}, max={overall['pass_at_1_rate']['max']:.3%}\n"
                f"Pass@k Rate     : mean={overall['pass_at_k_rate']['mean']:.3%}, min={overall['pass_at_k_rate']['min']:.3%}, max={overall['pass_at_k_rate']['max']:.3%}\n"
                f"Execution-rate  : mean={overall['execution_rate']['mean']:.3%}, min={overall['execution_rate']['min']:.3%}, max={overall['execution_rate']['max']:.3%}\n"
                f"Time cost (min) : mean={overall['duration_min']['mean']:.3f}, min={overall['duration_min']['min']:.3f}, max={overall['duration_min']['max']:.3f}\n"
                f"Token cost      : mean=${overall['token_cost']['mean']:.6f}, min=${overall['token_cost']['min']:.6f}, max=${overall['token_cost']['max']:.6f}\n"
                f"====================================================================================\n"
            )

            all_variant_overalls.append(
                {
                    "variant": variant_name,
                    "n_datasets": len(all_dataset_aggs),
                    "datasets": [d.get("dataset") for d in all_dataset_aggs],
                    "summary": overall,
                }
            )

            _append_to_results_log(
                {
                    "aggregate_all_datasets": True,
                    "ablation_tag": variant_name,
                    "n_datasets": len(all_dataset_aggs),
                    "datasets": [d.get("dataset") for d in all_dataset_aggs],
                    "summary": overall,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
                }
            )

    # Final: multi-ablation summary across variants
    if all_variant_overalls:
        def _vmean(metric_key: str) -> List[float]:
            vals = []
            for v in all_variant_overalls:
                s = v.get("summary") or {}
                m = (s.get(metric_key) or {}).get("mean", None)
                if m is not None:
                    vals.append(float(m))
            return vals

        overall_over_variants = {
            "pass_at_1_rate": _summarize(_vmean("pass_at_1_rate")),
            "pass_at_k_rate": _summarize(_vmean("pass_at_k_rate")),
            "execution_rate": _summarize(_vmean("execution_rate")),
            "duration_min": _summarize(_vmean("duration_min")),
            "token_cost": _summarize(_vmean("token_cost")),
        }

        print(
            f"\n================  MULTI-ABLATION SUMMARY over {len(all_variant_overalls)} variant(s)  ================\n"
            f"Pass@1(mean over datasets) : mean={overall_over_variants['pass_at_1_rate']['mean']:.3%}, min={overall_over_variants['pass_at_1_rate']['min']:.3%}, max={overall_over_variants['pass_at_1_rate']['max']:.3%}\n"
            f"Pass@k(mean over datasets) : mean={overall_over_variants['pass_at_k_rate']['mean']:.3%}, min={overall_over_variants['pass_at_k_rate']['min']:.3%}, max={overall_over_variants['pass_at_k_rate']['max']:.3%}\n"
            f"Execution(mean over datasets): mean={overall_over_variants['execution_rate']['mean']:.3%}, min={overall_over_variants['execution_rate']['min']:.3%}, max={overall_over_variants['execution_rate']['max']:.3%}\n"
            f"Time(min, mean over datasets): mean={overall_over_variants['duration_min']['mean']:.3f}, min={overall_over_variants['duration_min']['min']:.3f}, max={overall_over_variants['duration_min']['max']:.3f}\n"
            f"Cost($, mean over datasets)  : mean=${overall_over_variants['token_cost']['mean']:.6f}, min=${overall_over_variants['token_cost']['min']:.6f}, max=${overall_over_variants['token_cost']['max']:.6f}\n"
            f"------------------------------------------------------------------------------------\n"
            f"{'variant':<26} {'P@1(mean)':>10} {'P@k(mean)':>10} {'Exec(mean)':>11} {'Time(min)':>10} {'Cost($)':>10}\n"
            f"------------------------------------------------------------------------------------"
        )
        for v in all_variant_overalls:
            s = v.get("summary") or {}
            p1 = (s.get("pass_at_1_rate") or {}).get("mean", 0.0)
            pk = (s.get("pass_at_k_rate") or {}).get("mean", 0.0)
            ex = (s.get("execution_rate") or {}).get("mean", 0.0)
            tm = (s.get("duration_min") or {}).get("mean", 0.0)
            co = (s.get("token_cost") or {}).get("mean", 0.0)
            print(f"{str(v.get('variant')):<26} {p1:>10.3%} {pk:>10.3%} {ex:>11.3%} {tm:>10.3f} {co:>10.6f}")
        print("====================================================================================\n")

        # Append multi-ablation summary to log (best-effort)
        _append_to_results_log(
            {
                "aggregate_all_variants": True,
                "n_variants": len(all_variant_overalls),
                "variants": [v.get("variant") for v in all_variant_overalls],
                "summary": overall_over_variants,
                "variant_overalls": all_variant_overalls,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
            }
        )


if __name__ == "__main__":
    main()