import os
import time
import json
import copy
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.dataloader import DataLoader 
from src.utils import cal_time_cost, get_token_usage
from src.train_eval_utils import *
from src.llm_retriever import LibraryRetrieval

def run_library_diagnosis(
    iter, 
    train_tasks,
    llm_retri, llm_opt, llm_diag, llm_ins, library,
    params, 
    paths, 
    max_workers = 8,
):
    """
    Run library insight diagnosis through doing tasks
    """

    lock = Lock()               # Lock to safely update shared variables
    

    def _train_worker(task, train_output_path):
        """
        Parallelize the entire per-task pipeline (insight retrieval -> formulation generation -> insight retrieval -> program generation -> diagnosis -> insight extraction -> insight verification) for training tasks
        Return: (new insights, is_success, is_execution, is_verify, is_diagnosis)
        """
        nonlocal iter_self_verify_total, iter_self_verify_success_tasks, iter_self_verify_full_retrieval_tasks, iter_self_verify_partial_retrieval_tasks

        with lock:
            lib_snapshot = copy.deepcopy(library)
            taxo_snapshot = copy.deepcopy(library.taxonomy)

        llm_retri_local = LibraryRetrieval(
            lib=lib_snapshot,
            model=llm_retri.model,
            service=llm_retri.service,
            temperature=llm_retri.temp,
        )

        # For each task, create a temporary copy of the snapshot library state for new insight retrieval verification
        temp_library = copy.deepcopy(lib_snapshot)

        #* Retrieve insights (if any) and generate formulation and program
        prev_insights, candidate_formulation, candidate_program, output, runnable, is_time_out = generate_solution_with_retrieval(
                iter, task, lib_snapshot, llm_retri_local, llm_opt, 
                retrieved_insights=[],
                output_path=train_output_path, verbose=False, save_data=True
        )
        # Extreme case: code extraction failed
        if not candidate_program:
            task.output_status.append("parse_error")
            return [], False, False, None, None # new insights, is_success, is_execution, is_verify, is_diagnosis

        # Check optimality
        is_optimal, output_status, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)

        # Record task
        retrieved_ins_ids = [ins["insight_id"] for ins in prev_insights if "insight_id" in ins]             
        task.retri_ins_lst.append(retrieved_ins_ids)
        task.output_status.append(output_status)

        if is_optimal:
            print(f"\n   [Task {task.id}]: Output was optimal. Task succeeds!")
            return [], True, None, None, None  # new insights, is_success, is_execution, is_verify, is_diagnosis
        else:  
            print(feedback)

        # ============= Library Diagnosis ==============
        #* Diagnose the failure and retrieved insights (the cause of failure)
        is_need_formulation_diag = (output_status != "run_error")
        new_program_ins = []

        # First fix program that can not execute
        if output_status == "run_error":
            # === Step 1: diagnose retrieved program_ins, remove negative, regenerate program with original formulation ===
            formulation_ins, program_ins = divide_insight(prev_insights)
            original_program_ins = list(program_ins)

            print(f"[Task {task.id}] Calling diagnose_program (run_error branch) | orig_program_ins={len(original_program_ins)}")
            program_insights_diag = {}
            (
                program_insights_diag,
                program_ins,
                candidate_program,
                output,
                runnable,
                is_time_out,
                is_optimal,
                output_status,
                feedback,
            ) = llm_diag.diagnose_program(
                iter=iter,
                task=task,
                formulation=candidate_formulation,
                failed_program=candidate_program,
                feedback=feedback,
                retrieved_formulation_insights=formulation_ins,
                retrieved_program_insights=program_ins,
                llm_opt=llm_opt,
                llm_retri=llm_retri_local,
                max_unretrieved_trials=4,
                verbose=False,
                save_data=True,
                output_path=train_output_path,
            )

            # Track retrieval stats & distribution for program insights (pos/neg) based on originally retrieved ones.
            orig_prog_ids = [ins.get("insight_id") for ins in original_program_ins if ins.get("insight_id") is not None]
            orig_prog_ids = [iid for iid in orig_prog_ids if iid is not None]
            if orig_prog_ids:
                prog_positive_ids = [iid for iid in orig_prog_ids if program_insights_diag.get(iid) == "positive"]
                with lock:
                    library.update_retrieval_stats(orig_prog_ids, iter, success=False)
                    if prog_positive_ids:
                        library.update_retrieval_stats(prog_positive_ids, iter, success=True)

                    for ins in library:
                        if ins.insight_id in orig_prog_ids:
                            st = program_insights_diag.get(ins.insight_id, "positive")
                            if task.id not in ins.distribution.get(st, []):
                                ins.distribution.setdefault(st, []).append(task.id)

            # Also record any "unretrieved" program insights discovered during diagnosis (do not count as retrieved stats).
            unretrieved_ids = [iid for iid, st in program_insights_diag.items() if st == "unretrieved"]
            if unretrieved_ids:
                with lock:
                    for ins in library:
                        if ins.insight_id in unretrieved_ids:
                            if task.id not in ins.distribution.get("unretrieved", []):
                                ins.distribution.setdefault("unretrieved", []).append(task.id)

            prev_insights = formulation_ins + program_ins

            # If not run_error anymore, continue to formulation diagnosis (negative insights were the root cause).
            # If run_error has been resolved, we may already be optimal. In that case, skip further diagnosis.
            if is_optimal:
                print(f"\n   [Task {task.id}]: Output became optimal after fixing run_error. Task succeeds!")
                return [], True, None, None, None  # new insights, is_success, is_execution, is_verify, is_diagnosis

            # Otherwise, if it's runnable now but still not optimal, proceed to formulation diagnosis.
            if output_status != "run_error" and (not is_optimal):
                is_need_formulation_diag = True
            else:
                # === Step 3 (fallback): still run_error -> fix program to enable extracting NEW program insights ===
                is_optimal, runnable, corrected_program, _ = llm_diag.debug_program(
                    iter=iter,
                    task=task,
                    failed_program=candidate_program,
                    feedback=feedback,
                    verbose=False,
                    save_data=True,
                    output_path=train_output_path,
                )
                # If the fix fails, skip insight extraction for this task
                if not runnable:
                    task.fail_to_execute += 1
                    return [], False, False, None, None  # new insights, is_success, is_execution, is_verify, is_diagnosis

                print(f"\n   [Task {task.id}]: Succeeded to fix a program that failed to execution!")

                # Track best partial retrieval across attempts
                last_new_program_ins = None
                last_retrieval_result = None
                best_verified_insights = []

                for attempt_num in range(1, params.max_verify_attempts + 1):
                    # Decide whether to regenerate or to modify missed insights
                    modified_insight_ids = set()  # Track which insights were modified in this attempt
                    if last_retrieval_result is not None:
                        # Task was successful but retrieval was partial/none:
                        # skip regenerate and modify only missed insights to improve retrieval.
                        tax_failed_ids = last_retrieval_result.get("taxonomy_failed_insight_ids", set())
                        cond_failed_ids = last_retrieval_result.get("condition_failed_insight_ids", set())

                        new_program_ins = []
                        for ins in last_new_program_ins:
                            ins_id = ins.get("insight_id")
                            taxonomy_failed = ins_id in tax_failed_ids
                            # condition_failed only if taxonomy matched successfully but condition failed
                            condition_failed = ins_id in cond_failed_ids and ins_id not in tax_failed_ids
                            if taxonomy_failed or condition_failed:
                                modified_ins = llm_ins.modify_new_insight_for_retrieve(
                                    iter=iter,
                                    task=task,
                                    insight=ins,
                                    taxonomy_failed=taxonomy_failed,
                                    condition_failed=condition_failed,
                                    library=temp_library,
                                    candidate_formulation=candidate_formulation,
                                    verbose=False
                                )
                                new_program_ins.append(modified_ins)
                                modified_insight_ids.add(ins_id)  # Track original insight_id that was modified
                            else:
                                # Already retrievable insights are kept as-is
                                new_program_ins.append(ins)
                    else:
                        # Either first attempt or previous attempt had task failure:
                        # regenerate new program insights from corrected program.
                        new_program_ins = llm_ins.generate_insights(
                            iter=iter,
                            task=task,
                            corrected_program=corrected_program,
                            taxonomy=taxo_snapshot,
                            candidate_formulation=candidate_formulation,
                            verbose=False,
                            save_data=True,
                            output_path=train_output_path
                        )

                    if not new_program_ins:
                        # Nothing to verify in this attempt
                        continue

                    #* Keep only those new insights that can solve its source task when applied back and can be retrieved
                    # Combined verification: retrieval + task success
                    # Add new_program_ins to temp_library before verification
                    temp_library.add_insights_update_ids(new_program_ins, iter, lock=None)
                    
                    is_verify, verified_insights, task_success, retrieval_result = self_verify_retrieval_and_success(
                        iter=iter,
                        task=task,
                        llm_opt=llm_opt,
                        new_insights=new_program_ins,
                        prev_insights=prev_insights,
                        library=temp_library,
                        llm_retri=llm_retri_local,
                        candidate_formulation=candidate_formulation,
                        save_data=False,
                        output_path=train_output_path
                    )
                    with lock:
                        iter_self_verify_total += 1

                    # Remove unverified insights of the task from temp_library
                    temp_library.remove_unverified_insights(new_program_ins, is_verify, verified_insights, lock=None)

                    if not task_success:
                        # Case: task failed -> try regenerate on next attempt
                        last_new_program_ins = None
                        last_retrieval_result = None
                        continue

                    # From here on, task_success == True
                    last_new_program_ins = new_program_ins
                    last_retrieval_result = retrieval_result

                    if is_verify and verified_insights:
                        # Task success + full retrieval (full ⊂ success)
                        with lock:
                            iter_self_verify_success_tasks += 1
                            iter_self_verify_full_retrieval_tasks += 1
                        # Check if any modified insights were verified successfully
                        if modified_insight_ids:
                            verified_modified_ids = [v_ins.get("insight_id") for v_ins in verified_insights if v_ins.get("insight_id") in modified_insight_ids]
                            if verified_modified_ids:
                                print(f"✅ [Task {task.id}] Modified program insights were verified successfully at attempt {attempt_num}! Modified insight IDs: {verified_modified_ids}")
                        # Case 1: Task success + all insights retrieved
                        return verified_insights, False, True, True, None  # new insights, is_success, is_execution, is_verify, is_diagnosis
                    elif verified_insights:
                        # Case 2: Task success + partial retrieval - store for potential fallback
                        # Task success + partial retrieval (partial ⊂ success)
                        with lock:
                            iter_self_verify_success_tasks += 1
                            iter_self_verify_partial_retrieval_tasks += 1
                        # Check if any modified insights were verified successfully
                        if modified_insight_ids:
                            verified_modified_ids = [v_ins.get("insight_id") for v_ins in verified_insights if v_ins.get("insight_id") in modified_insight_ids]
                            if verified_modified_ids:
                                print(f"✅ [Task {task.id}] Modified program insights were verified successfully (partial) at attempt {attempt_num}! Modified insight IDs: {verified_modified_ids}")
                        # Track best partial insights across attempts
                        if len(verified_insights) > len(best_verified_insights):
                            best_verified_insights = verified_insights
                        # Do NOT break here; next attempt will refine missed insights
                        continue
                    else:
                        # Task success but no insights retrieved: rely on refinement in next attempts
                        continue

                # Max attempts reached (no full retrieval achieved)
                if best_verified_insights:
                    print(f"   [Task {task.id}]: Reached max attempts. Returning partial retrieval - {len(best_verified_insights)} insights")
                    return best_verified_insights, False, True, True, None

                # No verified program insights; proceed to formulation diagnosis if needed.
                is_need_formulation_diag = True


        if is_need_formulation_diag:
            # Convert insight
            formulation_ins, program_ins = divide_insight(prev_insights)
            # if isinstance(program_ins, dict):
            # print(" if is_need_formulation_diag: program_ins:", program_ins)
            prev_insights = [formulation_ins, program_ins]

            if new_program_ins:
                # if isinstance(new_program_ins, dict):
                # print("if new_program_ins: new_program_ins:", new_program_ins)
                #* If new program insights are generated, update them into the previous insights
                prev_insights[1] = new_program_ins

            #* Only when retrieved insights for formulation exist
            new_formulation = candidate_formulation
            updated_formulation_ins = []
            if  prev_insights[0]:
                insights_diag, updated_formulation_ins, is_generate_new, new_formulation = llm_diag.diagnose_formulation(
                    iter=iter,
                    task=task, 
                    feedback=feedback,
                    failed_formulation=candidate_formulation,
                    retrieved_insights=prev_insights,
                    llm_opt=llm_opt,
                    llm_retri=llm_retri_local,
                    verbose=True,
                    save_data=True,
                    output_path=train_output_path
                )  

                if insights_diag:
                    # Get all retrieved insight IDs for statistics
                    retrieved_ins_ids = list(insights_diag.keys())
                    # Get positive insight IDs (for correctness tracking)
                    positive_ins_ids = [iid for iid, state in insights_diag.items() if state == "positive"]
                    
                    # Update occurrence statistics for all retrieved insights
                    if retrieved_ins_ids:
                        with lock:
                            library.update_retrieval_stats(retrieved_ins_ids, iter, success=False)
                    
                    # Update correctness statistics for positive insights
                    if positive_ins_ids:
                        with lock:
                            library.update_retrieval_stats(positive_ins_ids, iter, success=True)
                    
                    #* For each retrieved insight, append its state with this task
                    with lock:
                        for ins in library:
                            if ins.insight_id in insights_diag.keys():
                                # Deduplicate {insight_id: unretrieved}
                                state = insights_diag[ins.insight_id]
                                if task.id not in ins.distribution[state]:
                                    ins.distribution[state].append(task.id)

            else: 
                is_generate_new = True

            if is_generate_new:
                # Track best partial retrieval across attempts
                last_new_formu_ins = None
                last_retrieval_result = None
                best_verified_insights = []

                for attempt_num in range(1, params.max_verify_attempts + 1):
                    # Decide whether to regenerate or to modify missed insights
                    modified_insight_ids = set()  # Track which insights were modified in this attempt
                    if last_retrieval_result is not None:
                        # Task was successful but retrieval was partial/none:
                        # skip regenerate and modify only missed insights to improve retrieval.
                        tax_failed_ids = last_retrieval_result.get("taxonomy_failed_insight_ids", set())
                        cond_failed_ids = last_retrieval_result.get("condition_failed_insight_ids", set())

                        new_formu_ins = []
                        for ins in last_new_formu_ins:
                            ins_id = ins.get("insight_id")
                            taxonomy_failed = ins_id in tax_failed_ids
                            # condition_failed only if taxonomy matched successfully but condition failed
                            condition_failed = ins_id in cond_failed_ids and ins_id not in tax_failed_ids
                            if taxonomy_failed or condition_failed:
                                modified_ins = llm_ins.modify_new_insight_for_retrieve(
                                    iter=iter,
                                    task=task,
                                    insight=ins,
                                    taxonomy_failed=taxonomy_failed,
                                    condition_failed=condition_failed,
                                    library=temp_library,
                                    candidate_formulation=candidate_formulation,
                                    verbose=False
                                )
                                new_formu_ins.append(modified_ins)
                                modified_insight_ids.add(ins_id)  # Track original insight_id that was modified
                            else:
                                # Already retrievable insights are kept as-is
                                new_formu_ins.append(ins)
                    else:
                        # Either first attempt or previous attempt had task failure:
                        # regenerate new formulation insights from failed formulation.
                        new_formu_ins = llm_ins.generate_insights(
                            iter=iter,
                            task=task,
                            failed_formulation=new_formulation,
                            taxonomy=taxo_snapshot,
                            verbose=False,
                            save_data=True,
                            output_path=train_output_path
                        )

                    if not new_formu_ins:
                        # Nothing to verify in this attempt
                        continue

                    if new_program_ins: 
                        new_insights = new_formu_ins + new_program_ins
                    else:
                        new_insights = new_formu_ins

                    #* Update the formulation insights with positive insights and unretrieved insights
                    prev_insights = [updated_formulation_ins, program_ins]
                    
                    prev_insights = prev_insights[0] + prev_insights[1]                    
                    # Add new_insights to temp_library before verification
                    temp_library.add_insights_update_ids(new_insights, iter, lock=None)
                    # Combined verification: retrieval + task success
                    is_verify, verified_insights, task_success, retrieval_result = self_verify_retrieval_and_success(
                        iter=iter,
                        task=task,
                        llm_opt=llm_opt,
                        new_insights=new_insights,
                        prev_insights=prev_insights,
                        library=temp_library,
                        llm_retri=llm_retri_local,
                        candidate_formulation=new_formulation,
                        save_data=False,
                        output_path=train_output_path
                    )
                    with lock:
                        iter_self_verify_total += 1

                    # Remove unverified insights from temp_library to maintain accurate context
                    temp_library.remove_unverified_insights(new_insights, is_verify, verified_insights, lock=None)

                    if not task_success:
                        # Case: task failed -> try regenerate on next attempt
                        last_new_formu_ins = None
                        last_retrieval_result = None
                        continue

                    # From here on, task_success == True
                    last_new_formu_ins = new_formu_ins
                    last_retrieval_result = retrieval_result

                    if is_verify and verified_insights:
                        # Task success + full retrieval (full ⊂ success)
                        with lock:
                            iter_self_verify_success_tasks += 1
                            iter_self_verify_full_retrieval_tasks += 1
                        # Check if any modified insights were verified successfully
                        if modified_insight_ids:
                            verified_modified_ids = [v_ins.get("insight_id") for v_ins in verified_insights if v_ins.get("insight_id") in modified_insight_ids]
                            if verified_modified_ids:
                                print(f"✅ [Task {task.id}] Modified formulation insights were verified successfully at attempt {attempt_num}! Modified insight IDs: {verified_modified_ids}")
                        # Case 1: Task success + all insights retrieved
                        print(f"The new generated insights of {task.id} are successfully verified!")
                        return verified_insights, False, True, True, False 
                    elif verified_insights:
                        # Case 2: Task success + partial retrieval - store for potential fallback
                        # Task success + partial retrieval (partial ⊂ success)
                        with lock:
                            iter_self_verify_success_tasks += 1
                            iter_self_verify_partial_retrieval_tasks += 1
                        # Check if any modified insights were verified successfully
                        if modified_insight_ids:
                            verified_modified_ids = [v_ins.get("insight_id") for v_ins in verified_insights if v_ins.get("insight_id") in modified_insight_ids]
                            if verified_modified_ids:
                                print(f"✅ [Task {task.id}] Modified formulation insights were verified successfully (partial) at attempt {attempt_num}! Modified insight IDs: {verified_modified_ids}")
                        # Track best partial insights across attempts
                        if len(verified_insights) > len(best_verified_insights):
                            best_verified_insights = verified_insights
                        # Do NOT break here; next attempt will refine missed insights
                        continue
                    else:
                        # Task success but no insights retrieved: rely on refinement in next attempts
                        continue

                # The for loop has no break statement (i.e., no full retrieval achieved)
                else:
                    # Fallback: if we ever had partial retrieval, keep the best partial insights
                    if best_verified_insights:
                        print(f"   [Task {task.id}]: Reached max attempts. Returning partial retrieval - {len(best_verified_insights)} insights")
                        return best_verified_insights, False, True, True, False
                    else:
                        return [], False, True, False, False

            else:
                # After insight diagnosis, No new insights are needed
                return [], False, True, None, True


    # Experiment metrics 
    # temp_lib = []
    train_success_flags = [False] * len(train_tasks)

    iter_diagnose_count = 0
    iter_diagnose_success = 0

    # Counters for self_verify_retrieval_and_success outcomes
    iter_self_verify_total = 0
    # "success" here means: task solved when applying new insights (either full or partial retrieval)
    iter_self_verify_success_tasks = 0
    iter_self_verify_full_retrieval_tasks = 0     # subset of success: task success + full retrieval
    iter_self_verify_partial_retrieval_tasks = 0  # subset of success: task success + partial retrieval

    fail_to_verify_lst = []
    fail_to_execute_lst = []


    # Initialize new_insights queue and lock for serial processing to avoid version conflicts
    from queue import Queue
    from queue import Empty
    import threading
    import time
    new_ins_queue = Queue()
    queue_lock = Lock()
    processing_active = True
    processed_count = 0
    
    # Counters for online merge rate calculation
    total_new_insights = 0
    successful_merges = 0


    def process_insights_queue():
        nonlocal processed_count, total_new_insights, successful_merges, processing_active
        # print("Start processing insights queue")
        while processing_active or not new_ins_queue.empty():
            try:
                new_insights = new_ins_queue.get(timeout=0.2)
            
            except Empty:
                # no work yet; loop again while producer is still active
                time.sleep(0.1)
                continue

            if not new_insights:
                new_ins_queue.task_done()
                continue

            for new_insight in new_insights:
                total_new_insights += 1

                # Take a consistent snapshot of the current main library for ALL read-only operations
                # in this online-merge attempt (avoid holding the lock across LLM calls).
                with lock:
                    lib_snapshot = copy.deepcopy(library)

                merged_insights, merged_task_to_iter, parent_ids = llm_ins.conduct_insight_online_merge(
                    new_insight=[new_insight],
                    library=lib_snapshot,
                    verbose=True
                )

                # If no merge occurred, add original insight directly
                if not merged_insights:
                    with lock:
                        library.add_insights([new_insight], iter)
                        library.update_taxonomy(new_insight)
                    print(f"No merge occurred for task {new_insight['task_id']} , adding original insight!")
                    continue

                # If merge occurred, verify merged insights
                merged_task_ids = merged_insights["task_id"] if merged_insights else []
                target_tasks = train_tasks.subset_by_ids(merged_task_ids)

                all_tasks_verified = True
                for task in target_tasks:
                    # Initialize prev_ins_ids for each task separately
                    prev_ins_ids = []
                    task_iter = merged_task_to_iter.get(task.id)
                    if task_iter == -1 and task.retri_ins_lst:
                        prev_ins_ids.extend(task.retri_ins_lst[-1])
                    elif task.retri_ins_lst and len(task.retri_ins_lst) > task_iter:
                        prev_ins_ids.extend(task.retri_ins_lst[task_iter])
                    elif task.retri_ins_lst:
                        prev_ins_ids.extend(task.retri_ins_lst[-1])
                        
                    prev_insights = lib_snapshot.retrieve_insights_by_id(prev_ins_ids) if prev_ins_ids else []

                    # Get insights for verification based on task_iter
                    if task_iter == -1:
                        # For new_insight's task_id: use all new insights for this task (excluding current) + merged_insights
                        # Handle both single task_id and list of task_ids (for merged insights)
                        task_new_insights = []
                        for ins in new_insights:
                            if ins != new_insight:
                                task_id = ins.get('task_id')
                                if isinstance(task_id, list):
                                    if task.id in task_id:
                                        task_new_insights.append(ins)
                                elif task_id == task.id:
                                    task_new_insights.append(ins)
                        task_new_insights.append(merged_insights)
                    else:
                        # For existing insights' task_id: use library insights from iteration 0, excluding merged ones
                        task_new_insights = []
                        for ins in lib_snapshot:

                            if ins.iteration == task_iter:
                                # Handle both single task_id and list of task_ids (for merged insights)
                                task_id = ins.task_id
                                if isinstance(task_id, list):
                                    if task.id in task_id:
                                        # Exclude only the parents that were actually merged
                                        if ins.insight_id not in parent_ids:
                                            task_new_insights.append(ins.to_dict())
                                elif task_id == task.id:
                                    # Exclude only the parents that were actually merged
                                    if ins.insight_id not in parent_ids:
                                        task_new_insights.append(ins.to_dict())
                        task_new_insights.append(merged_insights)
                    
                    is_verify = self_verify_test(iter=None, task=task, llm_opt=llm_opt,
                                                new_insights=task_new_insights, prev_insights=prev_insights)

                    if not is_verify:
                        all_tasks_verified = False
                        break

                if all_tasks_verified:
                    with lock:
                        # Parent insights might have been merged/removed by earlier online-merge operations.
                        # If any parent is missing, skip merge and keep parents untouched.
                        if parent_ids:
                            main_ids = {ins.insight_id for ins in library}
                            if not set(parent_ids).issubset(main_ids):
                                library.add_insights([new_insight], iter)
                                library.update_taxonomy(new_insight)
                                print(
                                    f"Online merge skipped for task {new_insight['task_id']} due to missing parent ids in main library. "
                                    f"Adding original insight!"
                                )
                                continue

                        # Compute merge_version from the parent insights being merged (depth-style: max(parent)+1).
                        parent_versions = []
                        if parent_ids:
                            # ExperienceLibrary supports sequence protocol (getitem/len); iterate to access Insight objects.
                            for _ins in library:
                                if _ins.insight_id in parent_ids:
                                    parent_versions.append(getattr(_ins, "merge_version", 0))
                        merged_insights["merge_version"] = (max(parent_versions) if parent_versions else 0) + 1
                        merged_insights["refine_version"] = 0

                        # If taxonomy is invalid (e.g., a plain string), skip this merged insight to avoid crashes/data loss.
                        taxo = merged_insights.get("taxonomy")
                        if not isinstance(taxo, dict):
                            # Try JSON-dict string
                            if isinstance(taxo, str) and taxo.strip().startswith("{") and taxo.strip().endswith("}"):
                                try:
                                    parsed = json.loads(taxo)
                                except Exception:
                                    parsed = None
                                if isinstance(parsed, dict):
                                    merged_insights["taxonomy"] = parsed
                            if not isinstance(merged_insights.get("taxonomy"), dict):
                                print(
                                    f"[WARNING] Merged insight has invalid taxonomy; skipping merged insight and keeping parents. "
                                    f"taxonomy_type={type(taxo).__name__}"
                                )
                                library.add_insights([new_insight], iter)
                                library.update_taxonomy(new_insight)
                                print(f"Online merge skipped for task {new_insight['task_id']} due to invalid taxonomy. Adding original insight!")
                                continue

                        # Remove only the existing insights that were actually merged (not all taxonomy-matched candidates).
                        library.replace_merged_insights([{"insight_id": pid} for pid in parent_ids])
                        library.add_insights([merged_insights], iter)
                        library.update_taxonomy(merged_insights)
                    successful_merges += 1
                    print(f"Successfully merged insight for task {new_insight['task_id']} with existing library insights (self-verify only)!")
                else:
                    with lock:
                        library.add_insights([new_insight], iter)
                        library.update_taxonomy(new_insight)
                    print(f"Online merge failed for task {new_insight['task_id']}. Adding original insight!")

                processed_count += 1
                if processed_count % 10 == 0:
                    try:
                        library_checkpoint = copy.deepcopy(library)
                        library_checkpoint.save(f"{paths.lib_dir}/library_iter{iter}_diag_snap.json")
                        library_checkpoint.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_iter{iter}_snap.json")
                        train_tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_iter{iter}_snap.json")
                        print(f"[Iteration {iter}] Saved library snapshot after processing {processed_count} insights")
                    except Exception as e:
                        print(f"[Iteration {iter}] Warning: Failed to save snapshot: {e}")

            new_ins_queue.task_done()

    # Start the async processing thread
    processing_thread = threading.Thread(target=process_insights_queue, daemon=True)
    processing_thread.start()

    # Training phase
    train_start_time = time.time()
    usage_before = get_token_usage()
    with ThreadPoolExecutor(max_workers=6) as executor:   #max_workers
        futures = {
            executor.submit(
                _train_worker,
                task,
                os.path.join(paths.train_output_dir, f"task_{task.id}"),     # per-task output folder
            ): (idx, task)
            for idx, task in enumerate(train_tasks)
        }

        for future in tqdm(as_completed(futures), total=len(train_tasks), desc=f"[Iteration {iter}] Library Diagnosis Phase\n"):
            idx, task = futures[future]
            new_insights, is_success, is_execution, is_verify, is_diagnosis = future.result()

            if is_execution is False:
                fail_to_execute_lst.append(task.id)

            if is_verify is False:
                fail_to_verify_lst.append(task.id)

            if is_diagnosis is not None:
                iter_diagnose_count += 1
                if is_diagnosis is True:
                    iter_diagnose_success += 1

            train_success_flags[idx] = is_success
            if new_insights:
                # Add new_insights to queue to avoid version conflicts during parallel merging
                # print("new_insights", new_insights)
                with queue_lock:
                    print('Start adding new insights')
                    new_ins_queue.put(new_insights)
                    print(f"Added new insights to queue! Now the queue has length {new_ins_queue.qsize()}")

    
    train_duration = cal_time_cost(train_start_time, f'Iteration {iter} Library Diagnosis Phase')

    # Stop the async processing and wait for queue to be empty
    print(f"[Iteration {iter}] Stopping async processing and waiting for queue to be empty...")
    processing_active = False
    
    # Wait for all items in queue to be processed
    print(f"[Iteration {iter}] Waiting for queue to be empty...")
    try:
        # Use join() with a simple timeout wrapper
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Queue join timeout")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)  # 30 minute timeout
        new_ins_queue.join()
        signal.alarm(0)  # Cancel the alarm
        print(f"[Iteration {iter}] Queue is now empty")
    except TimeoutError:
        print(f"[Iteration {iter}] WARNING: Queue join timeout after 30 minutes, forcing continue")
    except Exception as e:
        print(f"[Iteration {iter}] Error waiting for queue: {e}")
    
    # Wait for processing thread to finish with 5 minute timeout
    processing_thread.join(timeout=300)  # 5 minute timeout
    
    print(f"[Iteration {iter}] Queue processing completed, processed {processed_count} insights")
    
    # Update llm_retri with the latest library state
    llm_retri.library = library

    # ---- Token usage summary for this phase ----
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

    # Calculate the success rate for this iteration
    number_of_train_failures = len(train_success_flags) - sum(train_success_flags)
    train_accuracy = sum(train_success_flags) / len(train_success_flags) if train_success_flags else 0

    # Calculate detailed self-verify + retrieval statistics (based on self_verify_retrieval_and_success)
    self_verify_total = iter_self_verify_total if iter_self_verify_total > 0 else 1
    self_verify_task_success_rate = iter_self_verify_success_tasks / self_verify_total
    self_verify_full_retrieval_rate = iter_self_verify_full_retrieval_tasks / self_verify_total
    self_verify_partial_retrieval_rate = iter_self_verify_partial_retrieval_tasks / self_verify_total

    # Calculate task success rate after diagnosing and resolving problematic insights
    diagnose_success_rate = (iter_diagnose_success / iter_diagnose_count) if iter_diagnose_count > 0 else 0

    # Calculate online merge success rate: successfully merged insights / total new insights proposed
    online_merge_rate = (successful_merges / total_new_insights) if total_new_insights > 0 else 0

    # Record library learning success log
    iter_metrics = {
        "stage": "Library Diagnosis",
        "iter": iter,
        "train_accuracy": round(train_accuracy, 3),
        "library_size": len(library),
        "number_of_train_failures": number_of_train_failures,
        "self_verify_task_success_rate": round(self_verify_task_success_rate, 3),
        # 1) online merge accept/success counts / online merge proposed by LLM
        "online_merge_success_rate": round(online_merge_rate, 3),
        "online_merge_success_num": int(successful_merges),
        "online_merge_proposed_num": int(total_new_insights),
        # 2) solve tasks without generating new insights / diagnosis proposed from failed tasks
        "diagnosis_success_rate": round(diagnose_success_rate, 3),
        "diagnosis_success_num": int(iter_diagnose_success),
        "diagnosis_proposed_num": int(iter_diagnose_count),
        # 5/6) self-verify new-insight success counts / new-insight verify calls
        "self_verify_new_insight_full_success_rate": round(float(iter_self_verify_full_retrieval_tasks / iter_self_verify_total), 3) if iter_self_verify_total > 0 else 0,
        "self_verify_new_insight_full_success_num": int(iter_self_verify_full_retrieval_tasks),
        "self_verify_new_insight_partial_success_rate": round(float(iter_self_verify_partial_retrieval_tasks / iter_self_verify_total), 3) if iter_self_verify_total > 0 else 0,
        "self_verify_new_insight_partial_success_num": int(iter_self_verify_partial_retrieval_tasks),
        # Raw counters
        "iter_self_verify_total": iter_self_verify_total,
        "iter_diagnose_count": iter_diagnose_count,
        "total_new_insights": total_new_insights,
        "number_of_train_tasks": len(train_tasks) if train_tasks else 0,
        "fail_to_verify_task_ids": fail_to_verify_lst,
        "fail_to_execute_task_ids": fail_to_execute_lst,
        "library_diagnosis_duration (min)": train_duration,
        "token_usage": token_usage_delta,
    }

    return iter_metrics



if __name__ == "__main__":
    import os
    import time
    import json
    import copy

    from src.dataloader import DataLoader
    from src.utils import cal_time_cost
    from src.train_eval_utils import save_checkpoint
    from src.experience_library import ExperienceLibrary
    from src.llm_programmer import ProgramGenerator
    from src.llm_diagnostic import ProgramDiagnostic
    from src.llm_extractor import InsightExtractor
    from src.llm_retriever import LibraryRetrieval
    from src.llm_evolver import LibraryEvolution


    #* Configure
    from omegaconf import OmegaConf
    config = OmegaConf.load("train_config.yaml")

    #* Generate a timestamp and append it to output_folder
    # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # config.output_folder = f"{config.output_folder}_{ts}"
    # Re-resolve
    OmegaConf.resolve(config)

    # Initialize the LLM agents
    llm_opt = ProgramGenerator(model=config.base_model, service=config.service, temperature=0)
    llm_diag = ProgramDiagnostic(model=config.advanced_model, service=config.service, temperature=0)
    llm_ins = InsightExtractor(model=config.advanced_model, service=config.service, temperature=0.7)

    # 0 (start from online learning), 1 (start from library diagnosis at iter 1)
    start_iter = config.start_iter 
    # Load task recorded previously
    train_dataset_path = f"learning/{config.dataset}/train_tasks_record_iter{start_iter-1}.json"
    train_tasks = DataLoader(train_dataset_path, mode="learn", filter_success_num=None, reset=False)

    # Load previous library
    if start_iter == 1:
        library_path = f"{config.file_paths.lib_dir}/library_base.json"
    else:
        library_path = f"{config.file_paths.lib_dir}/library_refine_iter{start_iter-1}.json"
    library = ExperienceLibrary.from_json_file(
                                library_path = library_path,
                                taxonomy_path = f"{config.file_paths.lib_dir}/latest_taxonomy_iter{start_iter-1}.json")

    # Track iteration metrics
    with open(config.file_paths.metrics_log_path, "r") as f:
        metrics_log = json.load(f)

    # Run subset
    if config.data_slice:
        start = config.data_slice[0]
        end = config.data_slice[1]
        train_tasks = train_tasks.slice(start, end)

    start_time = time.time()

    #* Library Diagnosis
    llm_retri = LibraryRetrieval(lib=library, model=config.base_model, service=config.service, temperature=0)
    iter_metrics = run_library_diagnosis(
        start_iter, 
        train_tasks, 
        llm_retri, llm_opt, llm_diag, llm_ins, library, 
        config.params,
        config.file_paths,
        max_workers=8
    )

    # Save checkpoint
    metrics_log.append(iter_metrics)
    save_checkpoint(library=library, tasks=train_tasks, metrics=metrics_log, paths=config.file_paths, dataset=config.dataset, suffix=f"diag_iter{iter}")

    # Count time cost
    total_duration = cal_time_cost(start_time, f'The library diagnosis process')