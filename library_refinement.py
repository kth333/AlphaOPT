import json
import traceback
import copy
import time
import os
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import save_log_data, get_token_usage
from src.experience_library import ExperienceLibrary
from src.dataloader import DataLoader 
from src.llm_retriever import LibraryRetrieval
from src.llm_evolver import LibraryEvolution

from src.prompts.prompts_evolve import PROMPT_INS_REFINEMENT


def run_library_refinement(
    iter,
    tasks,
    config,
    llm_evolve,
    verbose=False,
    save_data=False,
    output_path=None,
    max_workers=4,
):
    """
    Parallelize only the outer loop over insights.
    """
    _start_time = time.time()

    def _process_one_insight(ins):
        """
        Process a single insight:
            1) Collect neg/unr reasons, call LLM to generate conditions per task.
            2) Integrate into a refinement prompt, get K new conditions.
            3) Build K library variants and verify retrieval to choose the best.
        Returns a dict with final condition and distribution to be merged in the main thread.
        """

        # Copy lists to avoid accidental in-place mutation on shared objects
        pos_task_ids = list(ins.distribution.get("positive") or [])
        neg_task_ids = list(ins.distribution.get("negative") or [])
        unr_task_ids = list(ins.distribution.get("unretrieved") or [])
        guard_unr_task_ids = list(ins.distribution.get("guard_unretrieved") or [])
        guard_neg_task_ids = list(ins.distribution.get("guard_negative") or [])
        # Tasks that were historically labeled "positive" but later became NOT retrieved after refinement.
        # We keep them for evaluation to track/penalize positive regressions across iterations.
        guard_lost_pos_task_ids = list(ins.distribution.get("guard_lost_positive") or [])

        def _dedup_keep_order(xs):
            seen = set()
            out = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        # If there are neither negative nor unretrieved tasks, skip refinement
        if not neg_task_ids and not unr_task_ids:
            return None

        # Add successful tasks that actually retrieved this insight into positive
        for task in tasks:
            if task.output_status[-1] == "optimal":
                if ins.insight_id in task.retri_ins_lst:
                    pos_task_ids.append(task.id)
        pos_task_ids = _dedup_keep_order(pos_task_ids)

        # Generate conditions for negative tasks
        neg_condition_lst = []
        for task in tasks.subset_by_ids(neg_task_ids):
            neg_condition = llm_evolve.generate_neg_condition(task, ins, iter, verbose=verbose, output_dir=output_path)
            neg_condition_lst.append(neg_condition)

        # Generate conditions for unretrieved tasks
        unr_condition_lst = []
        for task in tasks.subset_by_ids(unr_task_ids):
            unr_condition = llm_evolve.generate_unr_condition(task, ins, iter, verbose=verbose, output_dir=output_path)
            unr_condition_lst.append(unr_condition)

        #* Refine insight conditions 
        refined_conditions_k = llm_evolve.refine_insight(iter, neg_condition_lst, unr_condition_lst, ins, config.params.variant_num, verbose=verbose)

        # Build K library variants and evaluate
        library_variants_k = llm_evolve.build_library_variant(ins.insight_id, refined_conditions_k)

        # Evaluation task sets
        # Include guard sets in evaluation to avoid accepting changes that regress previously-fixed tasks.
        eval_pos_task_ids = _dedup_keep_order(pos_task_ids + guard_unr_task_ids + guard_lost_pos_task_ids)
        eval_neg_task_ids = _dedup_keep_order(neg_task_ids + guard_neg_task_ids)
        eval_unr_task_ids = _dedup_keep_order(unr_task_ids)

        total_tasks_num = len(eval_pos_task_ids + eval_neg_task_ids + eval_unr_task_ids)

        # Baseline performance BEFORE refinement.
        base_pos_retri_count = len(eval_pos_task_ids)
        base_neg_retri_count = len(_dedup_keep_order(neg_task_ids))
        base_unr_retri_count = 0

        base_performance = (base_pos_retri_count + base_unr_retri_count + len(eval_neg_task_ids) - base_neg_retri_count) / total_tasks_num if total_tasks_num > 0 else 0

        best_performance = base_performance
        best_pos_retri_count = base_pos_retri_count
        best_neg_retri_count = base_neg_retri_count
        best_unr_retri_count = base_unr_retri_count
        # Baseline "matched" sets derived from the same assumptions as base_*_retri_count.
        # This prevents accidentally treating everything as solved when no variant beats the baseline.
        best_matched_pos_tids = list(eval_pos_task_ids)  # assume all should-retrieve tasks are retrieved at baseline
        best_matched_neg_tids = list(_dedup_keep_order(neg_task_ids))  # assume active negatives are retrieved at baseline
        best_matched_unr_tids = []  # assume unretrieved tasks are NOT retrieved at baseline
        latest_condition = getattr(ins, "condition", None)

        # Decide which retrieval stage to use for this insight.
        # If the insight is a Code Implementation insight, it should be retrieved in the Program stage,
        # which requires a formulation (mathematical model) as context.
        ins_taxo = getattr(ins, "taxonomy", None) or {}
        ins_stage = "Program" if "Code Implementation" in ins_taxo else "Formulation"

        # For Program-stage verification, load the formulation from the previous round files:
        # learning/{output_folder}/task_{ID}/model_iter_{i}.txt
        # Cache per task-id to avoid repeated disk reads during variant evaluation.
        _formulation_cache = {}

        def _load_formulation_for_task(task):
            if task.id in _formulation_cache:
                return _formulation_cache[task.id]
            if not output_path:
                _formulation_cache[task.id] = None
                return None
            fp = os.path.join(str(output_path), f"task_{task.id}", f"model_iter_{iter}.txt")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    txt = f.read()
                txt = txt.strip() if isinstance(txt, str) else txt
                _formulation_cache[task.id] = txt
                return txt
            except Exception:
                _formulation_cache[task.id] = None
                return None

        formulation_lookup = _load_formulation_for_task if ins_stage == "Program" else None

        # Evaluate each variant
        for i, lib in enumerate(library_variants_k):
            llm_retri = LibraryRetrieval(lib=lib, model=config.base_model, service=config.base_service, temperature=0)
            pos_retri_count, matched_pos_tids = llm_evolve.verify_retrieval(
                ins.insight_id, tasks, eval_pos_task_ids, llm_retri,
                stage=ins_stage, config_override=config, formulation_lookup=formulation_lookup
            )
            neg_retri_count, matched_neg_tids = llm_evolve.verify_retrieval(
                ins.insight_id, tasks, eval_neg_task_ids, llm_retri,
                stage=ins_stage, config_override=config, formulation_lookup=formulation_lookup
            )
            # unr_retri_count means the number of tasks that have been retrieved
            unr_retri_count, matched_unr_tids = llm_evolve.verify_retrieval(
                ins.insight_id, tasks, eval_unr_task_ids, llm_retri,
                stage=ins_stage, config_override=config, formulation_lookup=formulation_lookup
            )

            # Variant scoring metric: the number of retrieved pos, unr insights and the decrease in neg insights number
            variant_performance = (pos_retri_count + unr_retri_count + len(eval_neg_task_ids) - neg_retri_count) / total_tasks_num if total_tasks_num > 0 else 0

            if variant_performance > best_performance:
                best_performance = variant_performance
                latest_condition = refined_conditions_k[i] if i < len(refined_conditions_k) else latest_condition
                best_pos_retri_count = pos_retri_count
                best_neg_retri_count = neg_retri_count
                best_unr_retri_count = unr_retri_count
                best_matched_pos_tids = matched_pos_tids
                best_matched_neg_tids = matched_neg_tids
                best_matched_unr_tids = matched_unr_tids

        performance_gain = best_performance - base_performance
        refinement_accepted = performance_gain > 0
        
        #* Update ins.distribution based on best variant results (new)
        # Remove solved active-negative tasks (no longer retrieved), but keep them in guard_negative
        # so that future refinements are evaluated against them (regression protection).
        solved_neg_tids = set(neg_task_ids) - (set(best_matched_neg_tids) & set(neg_task_ids))
        if solved_neg_tids:
            ins.distribution["negative"] = [
                tid for tid in ins.distribution.get("negative", [])
                if tid not in solved_neg_tids
            ]
            ins.distribution.setdefault("guard_negative", [])
            for tid in sorted(solved_neg_tids):
                if tid not in ins.distribution["guard_negative"]:
                    ins.distribution["guard_negative"].append(tid)
        
        # Remove solved active-unretrieved tasks (now retrieved), but keep them in guard_unretrieved
        # so that future refinements are evaluated against them (regression protection).
        solved_unr_tids = (set(best_matched_unr_tids) & set(unr_task_ids))
        if solved_unr_tids:
            ins.distribution["unretrieved"] = [
                tid for tid in ins.distribution.get("unretrieved", [])
                if tid not in solved_unr_tids
            ]
            # Store as "guard_unretrieved": previously unretrieved but fixed (now retrieved).
            ins.distribution.setdefault("guard_unretrieved", [])
            for tid in sorted(solved_unr_tids):
                if tid not in ins.distribution["guard_unretrieved"]:
                    ins.distribution["guard_unretrieved"].append(tid)

        # Move positive tasks that are no longer retrieved after refinement into guard_lost_positive.
        # This tracks positive regressions across iterations without mixing them into other buckets.
        active_pos_tids = set(ins.distribution.get("positive") or [])
        matched_active_pos_tids = set(best_matched_pos_tids) & active_pos_tids
        lost_pos_tids = active_pos_tids - matched_active_pos_tids
        if lost_pos_tids:
            ins.distribution["positive"] = [
                tid for tid in ins.distribution.get("positive", [])
                if tid not in lost_pos_tids
            ]
            ins.distribution.setdefault("guard_lost_positive", [])
            for tid in sorted(lost_pos_tids):
                if tid not in ins.distribution["guard_lost_positive"]:
                    ins.distribution["guard_lost_positive"].append(tid)
        
        # Return a compact result to be merged by the main thread
        return {
            "insight_id": ins.insight_id,
            "orig_condition": getattr(ins, "condition", None),
            "latest_condition": latest_condition,
            "distributions": {
                "positive": best_matched_pos_tids,
                "negative": best_matched_neg_tids,
                "unretrieved": best_matched_unr_tids
                },
            "performance_gain": performance_gain,
            "refinement_accepted": refinement_accepted,
            "report": (
                f"\nBest Performance on insight {ins.insight_id}: {best_performance} "
                f"\n Performance Gain: {performance_gain}"
                f"\npositive (before: {len(pos_task_ids)}; after: {best_pos_retri_count}) "
                f"\nnegative (before: {len(neg_task_ids)}; after: {best_neg_retri_count}) "
                f"\nunretrieved (before: {len(unr_task_ids)}; after: {len(unr_task_ids) - best_unr_retri_count})"
            )
        }

    # Results dictionaries (updated only in the main thread; no locks needed)
    refined_insights = {}         # {insight_id: [original_condition, refined_condition]}
    insight_distributions = {}    # {insight_id: {"positive": [...], "negative": [...], "unretrieved": [...]}}

    # Preselect insights to run for proper tqdm progress
    candidate_insights = [ins for ins in llm_evolve.library]
    total = len(candidate_insights)
    refined_ins_num = 0
    refinement_success_num = 0
    total_performance_gain = 0 

    # Track token usage before refinement
    usage_before = get_token_usage()

    # Thread pool over insights only
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=total, desc=f"[Iteration {iter}] Library Refinement") as pbar:
        future_map = {ex.submit(_process_one_insight, ins): ins.insight_id for ins in candidate_insights}
        for fut in as_completed(future_map):
            pbar.update(1)
            try:
                res = fut.result()
            except Exception:
                traceback.print_exc()
                continue
            # The insight do not have negatvie or unretrieved tasks
            if not res:
                continue

            refined_ins_num += 1 
            total_performance_gain += res["performance_gain"]
            if res.get("refinement_accepted"):
                refinement_success_num += 1
            iid = res["insight_id"]
            refined_insights[iid] = [res["orig_condition"], res["latest_condition"]]
            insight_distributions[iid] = res["distributions"]
            # Print once per completed insight to avoid interleaved outputs from threads
            print(res["report"])

    # persist results
    if save_data and output_path:
        refined_ins_list = [
            {
                "insight_id": insight_id,
                "original_condition": conds[0],
                "refined_condition": conds[1]
            }
            for insight_id, conds in refined_insights.items()
        ]
        save_log_data(refined_ins_list, f"{output_path}/refined_insights_iter{iter}.json")
        save_log_data(insight_distributions, f"{output_path}/refined_insight_distributions_iter{iter}.json")
    
    # Calculate the average refinement again (the average proportion of solved retrieval-misaligned tasks per insight)
    avg_refinement_rate = total_performance_gain / refined_ins_num if refined_ins_num else 0

    # Token usage summary for this phase
    usage_after = get_token_usage()
    token_usage_delta = {}
    for vendor, stats_after in usage_after.items():
        stats_before = usage_before.get(vendor, {})
        vendor_delta = {
            k: float(stats_after.get(k, 0.0) - stats_before.get(k, 0.0))
            for k in ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "cost")
        }
        # Only include vendors with non-zero cost
        if vendor_delta.get("cost", 0.0) != 0.0:
            token_usage_delta[vendor] = vendor_delta

    # Write refined conditions back to a copied library
    refined_library = copy.deepcopy(llm_evolve.library)
    for ins in refined_library:
        if ins.insight_id in refined_insights:
            ins.condition = refined_insights[ins.insight_id][1]
            # Increment refine_version for successfully refined insights
            ins.refine_version += 1
    # Attach token usage so caller can log if needed
    duration_min = (time.time() - _start_time) / 60.0
    # Additional refinement success metrics:
    # refined_ins_num: number of insights that needed refinement (had negative/unretrieved tasks)
    # refinement_success_num: number of insights whose refined variant was accepted (beat baseline)
    refinement_success_rate = (refinement_success_num / refined_ins_num) if refined_ins_num else 0
    return (
        refined_library,
        avg_refinement_rate,
        token_usage_delta,
        duration_min,
        refined_ins_num,
        refinement_success_rate,
        refinement_success_num,
    )


# Test a demo
if __name__ == "__main__":
    import time
    from datetime import datetime
    from src.utils import cal_time_cost
    from src.train_eval_utils import save_checkpoint

    #* Configure
    from omegaconf import OmegaConf
    config = OmegaConf.load("train_config.yaml")

    start_time = time.time()

    # Load previous library
    # 0 (start from online learning), 1 (start from library diagnosis at iter 1)
    start_iter = config.start_iter 
    library_path = f"{config.file_paths.lib_dir}/library_diag_iter{start_iter}.json"
    taxo_path = f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{start_iter}.json"
    library = ExperienceLibrary.from_json_file(
                                library_path = library_path,
                                taxonomy_path = taxo_path)

    # Load training data
    task_path = f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{start_iter}.json"
    train_tasks = DataLoader(task_path, mode="learn")

    # Library Evoluation
    llm_evolve = LibraryEvolution(lib=library, model=config.advanced_model, service=config.advanced_service, temperature=0.7)
    (
        refined_library,
        avg_refinement_rate,
        token_usage_delta,
        duration_min,
        refined_ins_num,
        refinement_success_rate,
        refinement_success_num,
    ) = run_library_refinement(
        iter=start_iter, tasks=train_tasks, 
        config=config, llm_evolve=llm_evolve,
        verbose=False, save_data=True, output_path=config.file_paths.train_output_dir,
        max_workers=8,
    )

    # Track iteration metrics
    with open(config.file_paths.metrics_log_path, "r") as f:
        metrics_log = json.load(f)

    last_metrics = metrics_log[-1]
    last_metrics["refinement_avg_gain"] = round(avg_refinement_rate, 3)
    last_metrics["refined_ins_num"] = int(refined_ins_num)
    # Requested metrics: accepted refinements / refinements attempted
    last_metrics["refinement_success_rate"] = round(float(refinement_success_rate), 3)
    last_metrics["refinement_success_num"] = int(refinement_success_num)
    last_metrics["refinement_proposed_num"] = int(refined_ins_num)
    last_metrics["refinement_token_usage"] = token_usage_delta
    last_metrics["library_refinement_duration (min)"] = round(float(duration_min), 3)

    
    print("refinement_avg_gain:", avg_refinement_rate)
    # Save library
    save_checkpoint(library=refined_library, tasks=None, metrics=metrics_log, paths=config.file_paths, suffix=f"refine_iter{start_iter}")

    # Count time cost
    total_duration = cal_time_cost(start_time, f'The library refinement process')