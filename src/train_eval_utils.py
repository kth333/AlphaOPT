import os
import re
import sys
import time
import json

import traceback
import subprocess
import numbers
from typing import List, Tuple, Optional, Any, Callable

from src.utils import cal_time_cost
from src.dataloader import DataLoader, Task          
from src.experience_library import ExperienceLibrary
from src.llm_retriever import LibraryRetrieval
from .utils import call_llm_and_parse_with_retry
import copy

#* Configure
from omegaconf import OmegaConf
config = OmegaConf.load("train_config.yaml")

def divide_insight(insight):
    # Divide insights into formulation and program stage
    formulation_ins = [
        ins for ins in insight
        if any(k in (ins.get("taxonomy") or {}) for k in ("Domain Modeling", "General Formulation"))
    ]
    program_ins = [ins for ins in insight if "Code Implementation" in ins.get("taxonomy", {})]

    return formulation_ins, program_ins


def generate_solution_with_retrieval(
            iter, task, library, llm_retri, llm_opt, 
            retrieved_insights=[],
            output_path="", verbose=False, save_data=True
            ):
    """
    Retrieve formulation insights -> Generate formulation -> Retrieve program insights -> Generate program
    Returns:
        candidate_formulation, program_output, runnable, is_time_out, retrieved_ins_ids
    """
    formulation_ins, program_ins = divide_insight(retrieved_insights)

    if not formulation_ins and any(key in ins.taxonomy for ins in library for key in ("General Formulation", "Domain Modeling")):
        # print(f"Retrieving formulation insights for Task {task.id}!")
        formulation_ins = llm_retri.retrieve_applicable_insights(
            iter=iter, task=task, stage="Formulation", config=config,
            verbose=verbose, save_data=save_data, output_path=output_path
        )
        # print(f"Retrieved {len(formulation_ins)} formulation insights for Task {task.id}!")

    # Generate mathematical formulation
    candidate_formulation = llm_opt.generate_formulation(
        iter=iter, task=task, retrieved_insights=formulation_ins, abl_params=config.ablation,
        verbose=verbose, save_data=save_data, output_path=output_path
    )

    if not candidate_formulation:
        return None, None, None, None, None, None

    # Retrieve insights for program generation
    if not program_ins and any("Code Implementation" in ins.taxonomy for ins in library):
        # print(f"Retrieving program insights for Task {task.id}!")
        program_ins = llm_retri.retrieve_applicable_insights(
            iter=iter, task=task, stage="Program", formulation=candidate_formulation, config=config,
            verbose=verbose, save_data=save_data, output_path=output_path
        )
        # print(f"Retrieved {len(program_ins)} program insights for Task {task.id}!")

    # Generate solver program
    candidate_program, output, runnable, is_time_out = llm_opt.generate_program(
        iter=iter, task=task, retrieved_insights=program_ins, formulation=candidate_formulation, abl_params=config.ablation,
        verbose=verbose, save_data=save_data, output_path=output_path
    )

    prev_insights = formulation_ins + program_ins

    return prev_insights, candidate_formulation, candidate_program, output, runnable, is_time_out


def is_optimal_with_tolerance(output, gt, tol=config.params.tolerance, mode="absolute"):

    if mode == "absolute":
        if abs(output - gt) <= tol:
            return True
        else:
            return False
    if mode == "relative":
        if abs(output - gt) <= tol * abs(gt):
            return True
        else:
            return False


def check_optimality(task, output, runnable, is_time_out):
    """
    Check if the output is optimal, non-optimal, or a failure to solve/run
    Returns
    -------
    (is_optimal: bool, status: str, feedback: str)
        - is_optimal : True iff output is numeric and within tolerance of ground_truth
        - status     : one of {"optimal", "not_optimal", "failure_solve", "solver_time_out", "run_error"}
        - feedback   : hints for code correction or debugging
    """

    # Accept broader numeric types (e.g., int, numpy scalars) to avoid misclassifying optimal outputs.
    # Exclude bool (a subclass of int) explicitly.
    if isinstance(output, numbers.Real) and not isinstance(output, bool):
        output = float(output)
        if is_optimal_with_tolerance(output=output, gt=task.ground_truth):
            return True, "optimal", None
        else:
            # Non-optimal results
            feedback = f"\n   [Task {task.id}]: Output was not optimal: {output}. Expected optimal value: {task.ground_truth}"
            return False, "not_optimal", feedback

    # No numeric objective returned
    if runnable:    
        if is_time_out:
            feedback = f"\n   [Task {task.id}]: Solver timed out without finding an optimal solution: \n{output}"
            return False, "solver_time_out", feedback

        feedback = f"\n   [Task {task.id}]: Failed to obtain an objective value: \n{output}"
        return False, "failure_solve", feedback

    # Program not runnable
    feedback = f"\n   [Task {task.id}]: Failed to generate a runnable program: \n{output}"
    return False, "run_error", feedback


# Verification function to check if newly added insights can be correctly retrieved
def verify_insight_retrieval(new_insights, library, target_task, llm_retri, iter=None, candidate_formulation: str | None = None):
    """
    Verify if newly added insights can be correctly retrieved by the retrieval system for a specific target task.
    """
    # Ensure new_insights is a list
    if not isinstance(new_insights, list):
        new_insights = [new_insights]
    
    if not new_insights:
        return
    
    # Create new retriever with temporary library
    temp_llm_retri = LibraryRetrieval(
        lib=library,
        model=llm_retri.model,
        service=llm_retri.service,
        temperature=llm_retri.temp
    )
    
    # Split new insights by stage based on taxonomy keys
    new_formulation_ins, new_program_ins = divide_insight(new_insights)

    def _ids(ins_list):
        return {ins.get("insight_id") for ins in (ins_list or []) if ins.get("insight_id") is not None}

    expected_formu = _ids(new_formulation_ins)
    expected_prog = _ids(new_program_ins)
    expected_insight_ids = expected_formu | expected_prog

    matched_ins_ids = set()
    applicable_ins_ids = set()

    # Step 1: Check taxonomy matching (Formulation + Program separately)
    if expected_formu:
        matched_formu = temp_llm_retri.quick_match_by_taxonomy(
            iter=iter,
            task=target_task,
            stage="Formulation",
            verbose=False,
            output_path=None
        )
        if matched_formu:
            matched_ins_ids |= {ins.get("insight_id") for ins in matched_formu if ins.get("insight_id") is not None}

    if expected_prog:
        matched_prog = temp_llm_retri.quick_match_by_taxonomy(
            iter=iter,
            task=target_task,
            stage="Program",
            formulation=candidate_formulation,
            verbose=False,
            output_path=None
        )
        if matched_prog:
            matched_ins_ids |= {ins.get("insight_id") for ins in matched_prog if ins.get("insight_id") is not None}

    taxonomy_failed_insights = expected_insight_ids - matched_ins_ids
    taxonomy_missed = len(taxonomy_failed_insights) > 0

    # Step 2: Check condition/applicability matching (Formulation + Program separately)
    if expected_formu:
        applicable_formu = temp_llm_retri.retrieve_applicable_insights(
            iter=iter,
            task=target_task,
            stage="Formulation",
            config=config,
            verbose=False,
            save_data=False,
            output_path=""
        )
        if applicable_formu:
            applicable_ins_ids |= {ins.get("insight_id") for ins in applicable_formu if ins.get("insight_id") is not None}

    if expected_prog:
        applicable_prog = temp_llm_retri.retrieve_applicable_insights(
            iter=iter,
            task=target_task,
            stage="Program",
            formulation=candidate_formulation,
            config=config,
            verbose=False,
            save_data=False,
            output_path=""
        )
        if applicable_prog:
            applicable_ins_ids |= {ins.get("insight_id") for ins in applicable_prog if ins.get("insight_id") is not None}

    retrieved_ins_ids = expected_insight_ids & applicable_ins_ids
    missed_insight_ids = expected_insight_ids - applicable_ins_ids
    applicability_missed = len(missed_insight_ids) > 0
    
    # Print verification results
    print(f"\n   [VERIFY RETRIEVAL] Verification Results for {len(new_insights)} new insight(s) on task {target_task.id}:")
    print(f"      Expected insight IDs: {expected_insight_ids}")
    print(f"      Retrieved insight IDs: {retrieved_ins_ids}")
    print(f"      Missed insight IDs: {missed_insight_ids}")
    print(f"      Taxonomy matched: {len(expected_insight_ids) - len(taxonomy_failed_insights)}/{len(expected_insight_ids)} insights")
    print(f"      Condition matched: {len(expected_insight_ids) - len(missed_insight_ids)}/{len(expected_insight_ids)} insights")
    
    all_retrieved = len(retrieved_ins_ids) == len(expected_insight_ids)
    
    if taxonomy_missed or applicability_missed:
        print(f"      ⚠️  WARNING: Some insights were not retrieved correctly!")
    else:
        print(f"      ✅ All insights were retrieved correctly on target task!")
    
    # Return verification results
    return {
        'all_retrieved': all_retrieved,
        'retrieved_insight_ids': retrieved_ins_ids,
        'missed_insight_ids': missed_insight_ids,
        'taxonomy_failed': taxonomy_missed,
        'condition_failed': applicability_missed,
        'taxonomy_failed_insight_ids': taxonomy_failed_insights,
        'condition_failed_insight_ids': missed_insight_ids
    }


def self_verify_test(iter, task, llm_opt, new_insights, prev_insights, save_data=False, output_path=""):
    # Combine new and previous insights
    prev_formulation_ins, prev_program_ins = divide_insight(prev_insights)
    new_formulation_ins, new_program_ins = divide_insight(new_insights)

    all_formulation_ins = prev_formulation_ins + new_formulation_ins
    all_program_ins = prev_program_ins + new_program_ins

    #* Call back and verify the effectiveness of relevant insights to the task
    candidate_formulation = llm_opt.generate_formulation(
        iter=iter,
        task=task,
        retrieved_insights=all_formulation_ins,
        abl_params=config.ablation,
        verbose=False,
        save_data=save_data,
        output_path=os.path.join(output_path, "self_verify")
    )

    _, output, runnable, is_time_out = llm_opt.generate_program(
        iter=iter,
        task=task,
        retrieved_insights=all_program_ins,
        formulation=candidate_formulation,
        abl_params=config.ablation,
        verbose=False,
        save_data=save_data,
        output_path=os.path.join(output_path, "self_verify")
    )

    # Check optimality with the same logic as the main pipeline (accept ints/numpy scalars, handle timeouts, etc.)
    is_optimal, _, _ = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
    return bool(is_optimal)


def self_verify_retrieval_and_success(
    iter, 
    task, 
    llm_opt, 
    new_insights, 
    prev_insights, 
    library, 
    llm_retri,
    candidate_formulation: str | None = None,
    save_data=False, 
    output_path=""
):
    """
    Combined verification: first check if insights can solve the task (self_verify_test),
    then check if they can be retrieved for the current task (verify_insight_retrieval).

    Returns:
        tuple: (is_verify: bool, verified_insights: list | None, task_success: bool, retrieval_result: dict | None)
            - is_verify:
                * True  : task success + all insights retrieved on target task (full retrieval)
                * False : otherwise
            - verified_insights:
                * If is_verify == True: original new_insights (all retrieved)
                * If is_verify == False and task success + partial retrieval: partial insights that were retrieved
                * If is_verify == False and task success + no retrieval: None
                * If is_verify == False and task failed: None
            - task_success: whether applying prev_insights + new_insights solved the task
            - retrieval_result: dict returned by verify_insight_retrieval (or None if not called / failed)
    """
    # Step 1: Self-verify test (can insights solve the task?)
    task_success = self_verify_test(
        iter=iter,
        task=task,
        llm_opt=llm_opt,
        new_insights=new_insights,
        prev_insights=prev_insights,
        save_data=save_data,
        output_path=output_path
    )
    
    if not task_success:
        # Case 4: Task failed
        return False, None, False, None

    # Step 2: Verify retrieval (only if task success)
    retrieval_result = verify_insight_retrieval(
        new_insights=new_insights,
        library=library,
        target_task=task,
        llm_retri=llm_retri,
        iter=iter,
        candidate_formulation=candidate_formulation,
    )
    
    # Default outputs (task_success is True here)
    is_verify = False
    verified_insights = None
    
    if retrieval_result is not None:
        # Check retrieval results
        all_retrieved = retrieval_result.get('all_retrieved', False)
        retrieved_insight_ids = retrieval_result.get('retrieved_insight_ids', set())
        
        if all_retrieved:
            # Case 1: Task success + retrieval on all insights
            is_verify = True
            verified_insights = new_insights
        elif retrieved_insight_ids:
            # Case 2: Task success + retrieval on partial insights
            # Filter insights by retrieved IDs
            filtered_insights = []
            for ins in new_insights:
                ins_id = ins.get('insight_id')
                if ins_id in retrieved_insight_ids:
                    filtered_insights.append(ins)
            verified_insights = filtered_insights if filtered_insights else None
        # else: Case 3: Task success + retrieval on no insights -> verified_insights remains None

    # If retrieval_result is None, treat as success but retrieval verification failed:
    # return is_verify=False, verified_insights=None, task_success=True, retrieval_result=None
    return is_verify, verified_insights, True, retrieval_result


def save_checkpoint(library, tasks, metrics, paths, suffix):
    os.makedirs(paths.lib_dir, exist_ok=True)
    os.makedirs(paths.train_output_dir, exist_ok=True)
    if library:
        # Save latest library and updated taxonomy
        library.save(f"{paths.lib_dir}/library_{suffix}.json")
        library.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_{suffix}.json")
    # Save tasks with status record
    if tasks:
        tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_{suffix}.json")
    if metrics:
        # Save iteration metrics log
        with open(paths.metrics_log_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


def print_training_metrics_summary(metrics_log):
    """
    Print structured summary of metrics from metrics_log.

    Args:
        metrics_log: List of dictionaries containing metrics for each iteration
    """
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)

    if not metrics_log:
        print("No metrics available.")
        return

    # Print header (rates + compact ratios)
    print(
        f"{'Iter':<4} {'Stage':<22} {'TrainAcc':<8} {'Fail':<6} {'Lib':<6} "
        f"{'OMerge':<18} {'SV-Merge':<18} {'Diag':<14} {'Refine':<16} "
        f"{'SV-Full':<16} {'SV-Part':<16}"
    )
    print("-" * 140)

    def format_value(value):
        if value == "N/A":
            return "N/A"
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)
        return str(value)

    def format_ratio(success_num, proposed_num, rate):
        if success_num == "N/A" or proposed_num == "N/A" or rate == "N/A":
            return "N/A"
        try:
            s = int(success_num)
            p = int(proposed_num)
        except Exception:
            return "N/A"
        if p <= 0:
            return f"{s}/{p}"
        try:
            r = float(rate)
            return f"{s}/{p}({r:.3f})"
        except Exception:
            return f"{s}/{p}"

    # Process each record in metrics_log (may be per-iteration or per-stage depending on pipeline)
    for i, metrics in enumerate(metrics_log):
        train_accuracy = metrics.get("train_accuracy", "N/A")
        number_of_train_failures = metrics.get("number_of_train_failures", "N/A")

        # Some stages use "library_size", others might not have it; keep fallback.
        library_size = metrics.get("library_size", "N/A")
        stage = metrics.get("stage", "N/A")

        # 1) Online merge success
        om_rate = metrics.get("online_merge_success_rate", "N/A")
        om_succ = metrics.get("online_merge_success_num", "N/A")
        om_prop = metrics.get("online_merge_proposed_num", "N/A")
        omerge = format_ratio(om_succ, om_prop, om_rate)

        # 4) Self-verify merged-group success
        svm_rate = metrics.get("self_verify_online_merge_success_rate", "N/A")
        svm_succ = metrics.get("self_verify_online_merge_success_num", "N/A")
        svm_prop = metrics.get("self_verify_online_merge_proposed_num", "N/A")
        svmerge = format_ratio(svm_succ, svm_prop, svm_rate)

        # 2) Diagnosis success (only meaningful for Diagnosis stage)
        diag_rate = metrics.get("diagnosis_success_rate", "N/A")
        diag_succ = metrics.get("diagnosis_success_num", "N/A")
        diag_prop = metrics.get("diagnosis_proposed_num", "N/A")
        diag = format_ratio(diag_succ, diag_prop, diag_rate) if diag_rate != "N/A" else "N/A"

        # 3) Refinement success (accepted refined variants)
        ref_rate = metrics.get("refinement_success_rate", "N/A")
        ref_succ = metrics.get("refinement_success_num", "N/A")
        ref_prop = metrics.get("refinement_proposed_num", metrics.get("refined_ins_num", "N/A"))
        refine = format_ratio(ref_succ, ref_prop, ref_rate)

        # 5/6) Self-verify new-insight (full/partial retrieval)
        sv_total = metrics.get("iter_self_verify_total", "N/A")
        sv_full_rate = metrics.get("self_verify_new_insight_full_success_rate", "N/A")
        sv_full_num = metrics.get("self_verify_new_insight_full_success_num", metrics.get("iter_self_verify_full_retrieval_tasks", "N/A"))
        sv_full = format_ratio(sv_full_num, sv_total, sv_full_rate)

        sv_part_rate = metrics.get("self_verify_new_insight_partial_success_rate", "N/A")
        sv_part_num = metrics.get("self_verify_new_insight_partial_success_num", metrics.get("iter_self_verify_partial_retrieval_tasks", "N/A"))
        sv_part = format_ratio(sv_part_num, sv_total, sv_part_rate)

        print(
            f"{i:<4} {str(stage)[:22]:<22} {format_value(train_accuracy):<8} {format_value(number_of_train_failures):<6} "
            f"{format_value(library_size):<6} {str(omerge)[:18]:<18} {str(svmerge)[:18]:<18} "
            f"{str(diag)[:14]:<14} {str(refine)[:16]:<16} {str(sv_full)[:16]:<16} {str(sv_part)[:16]:<16}"
        )

    print("=" * 140)

    # Print detailed breakdown (requested metrics)
    print("\nDETAILED BREAKDOWN (requested metrics):")
    print("-" * 60)

    for i, metrics in enumerate(metrics_log):
        stage = metrics.get("stage", "N/A")
        print(f"\nRecord {i} - {stage}:")

        # 1) Online merge
        print(f"  online_merge_success_rate: {metrics.get('online_merge_success_rate', 'N/A')}")
        print(f"  online_merge_success_num: {metrics.get('online_merge_success_num', 'N/A')}")
        print(f"  online_merge_proposed_num: {metrics.get('online_merge_proposed_num', 'N/A')}")

        # 4) Self-verify merged-group
        print(f"  self_verify_online_merge_success_rate: {metrics.get('self_verify_online_merge_success_rate', 'N/A')}")
        print(f"  self_verify_online_merge_success_num: {metrics.get('self_verify_online_merge_success_num', 'N/A')}")
        print(f"  self_verify_online_merge_proposed_num: {metrics.get('self_verify_online_merge_proposed_num', 'N/A')}")

        # 2) Diagnosis
        if "diagnosis_success_rate" in metrics or stage == "Library Diagnosis":
            print(f"  diagnosis_success_rate: {metrics.get('diagnosis_success_rate', 'N/A')}")
            print(f"  diagnosis_success_num: {metrics.get('diagnosis_success_num', 'N/A')}")
            print(f"  diagnosis_proposed_num: {metrics.get('diagnosis_proposed_num', 'N/A')}")

        # 3) Refinement
        if "refinement_success_rate" in metrics or "refined_ins_num" in metrics:
            print(f"  refinement_success_rate: {metrics.get('refinement_success_rate', 'N/A')}")
            print(f"  refinement_success_num: {metrics.get('refinement_success_num', 'N/A')}")
            print(f"  refinement_proposed_num: {metrics.get('refinement_proposed_num', metrics.get('refined_ins_num', 'N/A'))}")

        # 5/6) Self-verify new insights
        if "iter_self_verify_total" in metrics:
            print(f"  self_verify_new_insight_full_success_rate: {metrics.get('self_verify_new_insight_full_success_rate', 'N/A')}")
            print(f"  self_verify_new_insight_full_success_num: {metrics.get('self_verify_new_insight_full_success_num', 'N/A')}")
            print(f"  self_verify_new_insight_partial_success_rate: {metrics.get('self_verify_new_insight_partial_success_rate', 'N/A')}")
            print(f"  self_verify_new_insight_partial_success_num: {metrics.get('self_verify_new_insight_partial_success_num', 'N/A')}")

    print("\n" + "=" * 80)


def extract_code(text: str) -> str:
    """
    Extract a clean Python code snippet from the LLM output
    """
    code_block = None
    try:
        raw = text

        # Try to find a Markdown-style Python code block
        m = re.search(r"```python\s*\n([\s\S]*?)\n```", raw)
        if m:
            code_snippet = m.group(1).strip()
            code_block = m.group(0)  # for debugging
        else:
            # If no explicit Python fence, match any fenced code block
            m2 = re.search(r"```(?:\w*\s*)?\n([\s\S]*?)\n```", raw)
            if m2:
                code_snippet = m2.group(1).strip()
                code_block = m2.group(0)  # for debugging
            else:
                # If neither fence is present, raise an error
                raise ValueError(
                    "No valid code fence found. Expected a ```python``` block or a generic ``` block."
                )

        return code_snippet

    except Exception as e:
        print("LLM raw text:\n", text)
        print("Extracted code block:\n", code_block if code_block is not None else '<No code block>')
        print("Error during extract_code:", repr(e))
        raise


def execute_code(code_str, timeout_sec=400):
    try:
        # Using subprocess to execute the code as a separate process
        result = subprocess.run(
            [sys.executable, "-u", "-"], 
            input=code_str,
            text=True, 
            capture_output=True, 
            check=True,
            timeout=timeout_sec # Set the maximum run time
        )

        # Extract Gurobi's objVal (optimal objective value) from stdout
        output = result.stdout
        match = re.search(r"Optimal value\s*[:=]\s*([0-9.+-eE]+)", output)

        if match:
            solution = float(match.group(1))
            return solution
        else:
            return output
        
    except subprocess.TimeoutExpired as err:
        return err    
    

def self_debug(
    task: "Task" = None,
    failed_program: str = None,
    feedback: str = None,
    config: str = None
) -> Tuple[bool, Optional[str]]:           
    """
    Self-debug the failed program with LLM
    """
    runnable = False                    
    current_program  = failed_program
    current_feedback = feedback

    for attempt in range(1, config.ablation.max_debug_retry + 1):

        # Construct the prompt for diagnosis
        prompt = PROMPT_SELF_DEBUG.format(
            failed_program      = current_program,
            feedback            = current_feedback        
        )
        
        try:
            corrected_program = call_llm_and_parse_with_retry(
                model       = config.model,
                service     = config.service,
                prompt      = prompt,
                # Extract code script from LLM response
                parse_fn    = extract_code,
                temperature = 0,
                max_retry   = 3,                  
                sleep_sec   = 2,
                verbose     = False
            )

            # Update prompt context with new failed program
            current_program  = corrected_program

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to correct program\n")
            traceback.print_exc() # print error and cause
            return False, False

        #* Execute the corrected program
        try:
            output = execute_code(corrected_program) 
            runnable = True
            is_time_out = False
            #* Add solver time limitation to avoid large time cost on solving single task
            if isinstance(output, subprocess.TimeoutExpired):
                is_time_out = True
            else:
                try:
                    output = float(output) # ensure numerical outputs

                except (TypeError, ValueError):
                    pass # keep original output

            # Check optimality when the program is runnable
            is_optimal, _, current_feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
            return is_optimal, runnable

        except Exception as err:
            # Update prompt context with feedback about execution error
            current_feedback = f"Execution error:\n {err.stderr}"

    # Reached maximum retry for correction without successful execution
    is_optimal = False

    return is_optimal, runnable


def self_correction(
    task: "Task" = None,
    failed_formulation: str = None,
    failed_program: str = None,
    feedback: str = None,
    config: str = None
) -> Tuple[bool, Optional[str]]:           
    """
    Self-correct the failed formulation and program with LLM
    """
    runnable = False                    
    current_formulation  = failed_formulation
    current_program  = failed_program
    current_feedback = feedback

    for attempt in range(1, config.ablation.max_correction_retry + 1):

        # Construct the prompt for diagnosis
        prompt = PROMPT_SELF_CORRECTION.format(
            failed_formulation  = current_formulation,
            failed_program      = current_program,
            feedback            = current_feedback        
        )
        
        try:
            corrected_program = call_llm_and_parse_with_retry(
                model       = config.model,
                service     = config.service,
                prompt      = prompt,
                # Extract code script from LLM response
                parse_fn    = extract_code,
                temperature = 0,
                max_retry   = 3,                  
                sleep_sec   = 2,
                verbose     = False
            )

            # Update prompt context with new failed program
            current_program  = corrected_program

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to correct program\n")
            traceback.print_exc() # print error and cause
            return False, False

        #* Execute the corrected program
        try:
            output = execute_code(corrected_program) 
            runnable = True
            is_time_out = False
            #* Add solver time limitation to avoid large time cost on solving single task
            if isinstance(output, subprocess.TimeoutExpired):
                is_time_out = True
            else:
                try:
                    output = float(output) # ensure numerical outputs

                except (TypeError, ValueError):
                    pass # keep original output

            # Check optimality when the program is runnable
            is_optimal, _, current_feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
            return is_optimal, runnable

        except Exception as err:
            # Update prompt context with feedback about execution error
            current_feedback = f"Execution error:\n {err.stderr}"

    # Reached maximum retry for correction without successful execution
    is_optimal = False

    return is_optimal, runnable


PROMPT_SELF_DEBUG="""
You are an expert in Industrial Engineering and Operations Research. 

You are given:
1. A Gurobi program failed to execution
2. The execution error message for the failed program


### The failed program
{failed_program}


### Error message
{feedback}


### Your task
Your task is to review the execution error message, identify the issues in the failed program that caused the error, and revise the program so that it can run successfully.


### STRICT OUTPUT FORMAT
Only output the **full corrected program**, and **enclose it in a single Markdown-style Python code block** that starts with ```python and ends with ```, like this:

```python
import gurobipy as gp
from gurobipy import GRB
model = gp.Model("OptimizationProblem")
# your code starts from here
model.optimize()
```

- Ensure model.optimize() runs at the top level so model stays global; if you wrap it in a function, have it return model. Avoid any if __name__ == "__main__": guard.
- Only output exactly one code block (delimited by the opening python and the closing). Do not write any natural-language text outside the code block.
- **DO NOT MODIFY ANY CODE after the line model.optimize()**.

Now take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


PROMPT_SELF_CORRECTION="""
You are an expert in Industrial Engineering and Operations Research.

You are given:
1) the problem description of the optimization task,
2) the mathematical formulation your colleague proposed,
3) the Gurobi program based on the formulation,
4) the output (and possibly the error message) of executing the Gurobi program.

### Problem description
{problem_description}

### Mathematical formulation
{candidate_formulation}

### The Gurobi program
{candidate_program}

### Execution Output
{feedback}

### Your task
Carefully review the problem description, the proposed mathematical formulation, the Gurobi program, and the execution output. Determine whether BOTH the formulation AND the program are correct and faithful to the problem description.

If BOTH are correct: explain briefly why (the “reason”) and mark the status as "correct".
If EITHER is incorrect: provide an “analysis” that pinpoints the issues (modeling mismatch, wrong objective sign, missing/incorrect constraints, indexing/domain errors, integrality, parameter usage, solver API misuse, etc.), give a concise “reason”, and then provide a fully corrected version of the formulation and/or program.

### STRICT OUTPUT FORMAT (JSON ONLY)
Return a SINGLE JSON object with the following fields and NOTHING else (no Markdown fences, no extra text):

{
  "status": "correct" | "incorrect",
  "reason": "<1-3 sentence justification>",
  "analysis": "<deeper diagnosis; REQUIRED if status=='incorrect', otherwise may be empty>",
  "corrected_formulation": "<revised formulation in clear math/LaTeX/plaintext, or null if no change>",
  "corrected_program": "<FULL runnable Python program as a single string with \\n for newlines, or null if no change>",
  "change_log": ["<bullet point of a concrete change>", "..."]
}

#### Program requirements (when providing "corrected_program"):
- Provide a COMPLETE runnable Python script compatible with Gurobi (e.g., imports, model creation, variables, constraints, objective, model.optimize()).
- Include `model.optimize()` at top level (not gated by `if __name__ == '__main__':`).
- Do NOT include any text outside the Python code in the string.
- Do NOT include Markdown code fences inside the string.
- Do NOT place any code AFTER the line containing `model.optimize()`.

#### Additional guidance
- Judge correctness against the problem description first; use execution output for clues (e.g., infeasibility, domain/index errors, attribute errors).
- If the formulation is sound but the code has API or indexing bugs, mark status "incorrect" and fix the code.
- If the formulation is flawed, repair both the formulation and the code to match the problem description.
- Prefer explicit index sets, variable domains, and constraint names for clarity.
- Keep the “reason” concise and the “analysis” actionable.

Think step by step, but OUTPUT ONLY the final JSON object.
"""


def self_correction(ques, five, code, output, error):
    return f"""For the following optimization problem, modeling is performed, and pyomo code is generated and executed based on the modeling. Please judge whether the modeling and code are correct.
        The problem is as follows.

        {ques}

        The five-element formulation is as follows.

        {five}

        The code is as follows.

        {code}

        Run the code and get the following running information. 

        {output}
        {error}

        Please judge whether the above five-element and code are correct, and give your analysis according to the template below.

        ```
        The five-element is [Fill in True/False here].

        The code is [Fill in True/False here].

        Analysis:
        [Fill in your analysis here]
        ```"""