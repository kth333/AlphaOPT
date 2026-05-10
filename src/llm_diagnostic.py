import os
import re
import sys
import json
import subprocess
from typing import Optional, List, Tuple
import traceback
import itertools

from .utils import save_log_data, extract_json_array, call_llm_and_parse_with_retry
from .dataloader import DataLoader, Task
from .llm_programmer import ProgramGenerator
from .llm_retriever import LibraryRetrieval
from .train_eval_utils import check_optimality, is_optimal_with_tolerance

from .prompts.prompts_diag import (
    PROMPT_DIAGNOSE_ISSUES,
    PROMPT_INS_POS_NEG,
    PROMPT_VALIDATE_ISSUES,
    PROMPT_PROGRAM_DIAG,
    PROMPT_PROGRAM_INS_POS_NEG,
)


#* Configure
from omegaconf import OmegaConf
config = OmegaConf.load("train_config.yaml")

class ProgramDiagnostic:
    """
    LLM_diag agent: Provide diagnoses and code corrections from failed programs
    """
    def __init__(self, model: str, service: str, temperature: float | None = None):
        self.model = model
        self.service = service
        self.temp = temperature

    
    def extract_code(self, text: str) -> str:
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


    def execute_code(self, file_path, timeout_sec=400):
        try:
            # Using subprocess to execute the code as a separate process
            result = subprocess.run(
                [sys.executable, file_path], 
                capture_output=True, 
                text=True, 
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


    def _diagnose_issues(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_formulation: str = None,
        feedback: str = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ):           
        """
        Diagnose the issues in the failed formulation
        """

        # Construct the prompt for diagnosis
        prompt = PROMPT_DIAGNOSE_ISSUES.format(
            problem_description=task.desc,
            failed_formulation=failed_formulation,
            feedback=feedback,
            correct_program=task.correct_program,
        )

        # Call the LLM to generate the answer and extract code from string 
        log_header = (f"\n==========\n[Iteration {iter}] Diagnose the issues in Task {task.id}\n==========\n")
        error_message = f"\n   Task {task.id} failed to diagnose issues from LLM after maximum attempts\n"
        
        try:
            diagnosed_issues = call_llm_and_parse_with_retry(
                model       = self.model,
                service     = self.service,
                prompt      = prompt,
                # Extract code script from LLM response
                parse_fn    = extract_json_array,
                temperature = self.temp,
                max_retry   = 5,                  
                sleep_sec   = 2,
                verbose     = verbose,
                log_header  = log_header,
                error_message = error_message
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to diagnose issues\n")
            traceback.print_exc()
            return {}

        if save_data:
            # Save and run corrected code
            issues_diag_path = f"{output_path}/Diagnosis/issues_diagnosis_iter_{iter}.json"
            save_log_data(diagnosed_issues, issues_diag_path)

        return diagnosed_issues


    def _diagnose_pos_neg(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_formulation: str = None,
        diagnosed_issues: List[dict] = [],
        retrieved_insights: List[dict] = [],
        llm_opt: "ProgramGenerator" = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ):           
        """
        Diagnose the state of retrieved insights (positive or negative)
        """
        formulation_ins, program_ins = retrieved_insights

        # Construct the prompt for diagnosis
        prompt = PROMPT_INS_POS_NEG.format(
            problem_description=task.desc,
            failed_formulation=failed_formulation,
            diagnosed_issues=json.dumps(diagnosed_issues),
            retrieved_insights=formulation_ins,
        )

        # Call the LLM to generate the answer and extract code from string 
        log_header = (f"\n==========\n[Iteration {iter}] Diagnose the failed mathematical formulation for Task {task.id}\n==========\n")
        error_message = f"\n   Task {task.id} failed to diagnose mathematical formulation from LLM after maximum attempts\n"
        
        try:
            insights_diag = call_llm_and_parse_with_retry(
                model       = self.model,
                service     = self.service,
                prompt      = prompt,
                # Extract code script from LLM response
                parse_fn    = extract_json_array,
                temperature = self.temp,
                max_retry   = 5,                  
                sleep_sec   = 2,
                verbose     = verbose,
                log_header  = log_header,
                error_message = error_message,
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to diagnose insights\n")
            traceback.print_exc() # print error and cause
            # Return default values matching the expected return signature:
            # (insights_diag, pos_formulation_ins, is_retrieve_new, new_formulation, updated_issues)
            formulation_ins, program_ins = retrieved_insights
            return {}, formulation_ins, True, failed_formulation, diagnosed_issues

        # Validate insights_diag is a list
        if not isinstance(insights_diag, list) or not insights_diag:
            print(f"\n   [WARNING] Task {task.id}: Invalid insights_diag format, using default values\n")
            formulation_ins, program_ins = retrieved_insights
            return {}, formulation_ins, True, failed_formulation, diagnosed_issues

        if save_data:
            # Save and run corrected code
            insights_diag_path = f"{output_path}/Diagnosis/ins_pos_neg_diagnosis_iter_{iter}.json"
            save_log_data(insights_diag, insights_diag_path)

        # [{"insight_id":1, "state":"positive"}, ...]
        insights_diag = [{"insight_id": ins["insight_id"], "state": ins["state"]} for ins in insights_diag]

        if all(ins.get("state") in ("positive", "invalid") for ins in insights_diag):
            # It is necessary to generate new insights
            is_retrieve_new = True
            pos_formulation_ins = formulation_ins
            new_formulation = failed_formulation
            updated_issues = diagnosed_issues

        else:
            # Exclude the misleading insights and try to generate formulation and program again
            pos_ins_ids = [ins["insight_id"] for ins in insights_diag if ins["state"] == "positive"]
            pos_formulation_ins = [ins for ins in formulation_ins if ins["insight_id"] in pos_ins_ids]

            new_formulation = llm_opt.generate_formulation(
                iter=iter,
                task=task,
                retrieved_insights=pos_formulation_ins,
                # rewrite=False,
                abl_params=config.ablation,
                verbose=False,
                save_data=True,
                output_path=output_path
            )

            _, output, runnable, is_time_out = llm_opt.generate_program(
                iter=iter,
                task=task,
                retrieved_insights=program_ins,
                formulation=new_formulation,
                abl_params=config.ablation,
                verbose=False,
                save_data=True,
                output_path=output_path
            )

            if save_data:
                formu_path = f"{output_path}/Diagnosis/formu1_iter_{iter}.py"
                save_log_data(new_formulation, formu_path)

            # Check optimality
            is_optimal, _, feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
            if is_optimal:
                # It is not necessary to generate new insights
                is_retrieve_new = False 
                updated_issues = None

            else:
                is_retrieve_new = True 

                #* Diagnose the issues in the new formulation again after removing negative insights
                updated_issues = self._diagnose_issues(
                    iter=iter,
                    task=task,
                    failed_formulation=new_formulation,
                    feedback=feedback,
                    verbose=False,
                )

        insights_diag = {ins["insight_id"]: ins["state"] for ins in insights_diag}

        return insights_diag, pos_formulation_ins, is_retrieve_new, new_formulation, updated_issues

    def diagnose_program(
        self,
        *,
        iter: int = None,
        task: "Task" = None,
        formulation: str = None,
        failed_program: str = None,
        feedback: str = None,
        retrieved_formulation_insights: Optional[List[dict]] = None,
        retrieved_program_insights: List[dict] = None,
        llm_opt: Optional["ProgramGenerator"] = None,
        llm_retri: Optional["LibraryRetrieval"] = None,
        max_unretrieved_trials: int = 4,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ) -> tuple[dict, list[dict], Optional[str], object, bool, bool, bool, str, Optional[str]]:
        """
        Diagnose and fix run_error by:
          1) labeling retrieved program-stage insights as positive/negative and filtering "negative"
          2) regenerating a program with filtered insights
          3) if still run_error, retrieving extra candidate program insights and trying a few ("unretrieved")
             - insights that resolve run_error are labeled as "unretrieved"

        Returns:
          - insights_diag: {insight_id: state}
          - final_program_insights: filtered list (and possibly augmented with one "unretrieved" insight)
          - candidate_program, output, runnable, is_time_out, is_optimal, output_status, feedback
        """
        if llm_opt is None:
            raise ValueError("diagnose_program requires llm_opt (ProgramGenerator) to regenerate/check programs.")

        retrieved_formulation_insights = retrieved_formulation_insights or []
        retrieved_program_insights = retrieved_program_insights or []
        original_program_insights = list(retrieved_program_insights)

        # ===== Signals =====
        # 1) Start program_ins diagnosis
        try:
            tid = task.id
        except Exception:
            tid = None
        print(
            f"[Task {tid}]: Start diagnosing program insights"
        )

        insights_diag: dict = {}
        filtered: list[dict] = []
        had_negative = False
        used_unretrieved = False

        if retrieved_program_insights:
            prompt = PROMPT_PROGRAM_INS_POS_NEG.format(
                problem_description=task.desc,
                mathematical_model=formulation or "",
                failed_program=failed_program or "",
                feedback=feedback or "",
                retrieved_insights=json.dumps(retrieved_program_insights, indent=2, ensure_ascii=False),
            )

            log_header = (
                f"\n==========\n[Iteration {iter}] Diagnose program insights (pos/neg) for Task {task.id}\n==========\n"
            )
            error_message = (
                f"\n   Task {task.id} failed to diagnose program insights after maximum attempts\n"
            )

            try:
                diag_list = call_llm_and_parse_with_retry(
                    model=self.model,
                    service=self.service,
                    prompt=prompt,
                    parse_fn=extract_json_array,
                    temperature=self.temp,
                    max_retry=3,
                    sleep_sec=2,
                    verbose=verbose,
                    log_header=log_header,
                    error_message=error_message,
                )
            except Exception:
                # Fallback: keep all insights as positive (no filtering)
                diag_list = [{"insight_id": ins.get("insight_id"), "state": "positive"} for ins in retrieved_program_insights]

            # Normalize and validate
            if isinstance(diag_list, list):
                for item in diag_list:
                    if not isinstance(item, dict):
                        continue
                    iid = item.get("insight_id")
                    state = item.get("state")
                    if iid is None:
                        continue
                    if state not in ("positive", "negative"):
                        # Default unknown states to positive to be conservative
                        state = "positive"
                    insights_diag[iid] = state

            # Ensure every retrieved insight_id has a label
            for ins in retrieved_program_insights:
                iid = ins.get("insight_id")
                if iid is None:
                    continue
                insights_diag.setdefault(iid, "positive")

            if save_data:
                diag_path = f"{output_path}/Diagnosis/program_ins_pos_neg_diagnosis_iter_{iter}.json"
                save_log_data(
                    [{"insight_id": k, "state": v} for k, v in insights_diag.items()],
                    diag_path,
                )

            filtered = [
                ins for ins in retrieved_program_insights
                if insights_diag.get(ins.get("insight_id"), "positive") != "negative"
            ]
            had_negative = any(st == "negative" for st in insights_diag.values())

        # Re-generate program with filtered (non-negative) program insights
        candidate_program, output, runnable, is_time_out = llm_opt.generate_program(
            iter=iter,
            task=task,
            retrieved_insights=filtered,
            formulation=formulation,
            abl_params=config.ablation,
            verbose=False,
            save_data=True,
            output_path=output_path,
        )
        is_optimal, output_status, new_feedback = check_optimality(
            task=task, output=output, runnable=runnable, is_time_out=is_time_out
        )
        # Prefer the most recent feedback if available
        feedback = new_feedback or feedback

        final_program_insights = list(filtered)

        # 2) negative removed & runnable after regeneration
        if had_negative and output_status != "run_error":
            print(
                f"[Task {tid}]: Found negative program insights, removed and now runnable."
            )

        # If still run_error, try to find unretrieved program insights and retry program generation.
        if output_status == "run_error" and llm_retri is not None:
            exclude_ids = {
                ins.get("insight_id")
                for ins in (retrieved_formulation_insights + original_program_insights)
                if ins.get("insight_id") is not None
            }
            extra_program_ins = llm_retri.retrieve_applicable_insights(
                iter=iter,
                task=task,
                stage="Program",
                formulation=formulation,
                config=config,
                verbose=False,
                save_data=True,
                output_path=output_path,
            )
            extra_program_ins = [
                ins for ins in (extra_program_ins or [])
                if ins.get("insight_id") is not None and ins.get("insight_id") not in exclude_ids
            ]

            for cand in extra_program_ins[:max_unretrieved_trials]:
                trial_program_ins = final_program_insights + [cand]
                trial_program, trial_output, trial_runnable, trial_is_time_out = llm_opt.generate_program(
                    iter=iter,
                    task=task,
                    retrieved_insights=trial_program_ins,
                    formulation=formulation,
                    abl_params=config.ablation,
                    verbose=False,
                    save_data=True,
                    output_path=output_path,
                )
                trial_is_optimal, trial_output_status, trial_feedback = check_optimality(
                    task=task,
                    output=trial_output,
                    runnable=trial_runnable,
                    is_time_out=trial_is_time_out,
                )

                if trial_output_status != "run_error":
                    iid = cand.get("insight_id")
                    if iid is not None:
                        insights_diag[iid] = "unretrieved"
                    used_unretrieved = True

                    final_program_insights = trial_program_ins
                    candidate_program = trial_program
                    output = trial_output
                    runnable = trial_runnable
                    is_time_out = trial_is_time_out
                    is_optimal = trial_is_optimal
                    output_status = trial_output_status
                    feedback = trial_feedback or feedback
                    # 3) unretrieved added & runnable
                    print(
                        f"[Task {tid}]: Found unretrieved program insight, added and now runnable."
                    )
                    break

            if save_data and any(st == "unretrieved" for st in insights_diag.values()):
                unretrieved_path = f"{output_path}/Diagnosis/program_unretrieved_iter_{iter}.json"
                save_log_data(
                    [{"insight_id": k, "state": v} for k, v in insights_diag.items() if v == "unretrieved"],
                    unretrieved_path,
                )

        # 4) Still need new insights (either still run_error, or runnable but not optimal)
        if output_status == "run_error":
            print(f"[Task {tid}] 4 Need new insights | reason=still_run_error")
        elif not is_optimal:
            # runnable but not optimal -> likely proceed to formulation diagnosis / new insights
            # (if it became runnable due to negative removal or unretrieved addition, 2/3 signals already printed)
            print(f"[Task {tid}] 4 Need new insights | reason=not_optimal")

        return (
            insights_diag,
            final_program_insights,
            candidate_program,
            output,
            runnable,
            is_time_out,
            is_optimal,
            output_status,
            feedback,
        )

    def _get_insight_id(self, ins) -> Optional[str]:
        """
        Robustly extract insight_id from either a dict-like insight or an object.
        Returns None if insight_id is missing.
        """
        if ins is None:
            return None
        if isinstance(ins, dict):
            return ins.get("insight_id")
        return getattr(ins, "insight_id", None)

    def _dedup_inner_list_by_id(self, lst):
        "Deduplicate a list by insight_id (keep the first occurrence)."
        seen = set()
        out = []
        for ins in lst:
            iid = self._get_insight_id(ins)
            # If insight_id is missing, keep it (treat as unique).
            if iid is None or iid not in seen:
                if iid is not None:
                    seen.add(iid)
                out.append(ins)
        return out

    def unique_combo_gen(self, lists_of_ins):
        """
        Generate unique insight combinations from Cartesian product:
        - dedup within a combo by insight_id (keep first occurrence)
        - dedup across combos by sorted insight_id signature
        """
        seen_keys = set()
        for combo in itertools.product(*lists_of_ins):
            # Dedup within a combo by insight_id (e.g., (A,B,A) -> [A,B]), keep first occurrence.
            uniq = []
            seen_ids = set()
            for ins in combo:
                iid = self._get_insight_id(ins)
                if iid is None:
                    uniq.append(ins)
                    continue
                if iid not in seen_ids:
                    seen_ids.add(iid)
                    uniq.append(ins)

            # Use a sorted insight_id signature to avoid cross-combo duplicates (e.g., (A,B) vs (B,A,A)).
            key = tuple(sorted(seen_ids))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            yield uniq

    def _diagnose_unretrieved(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_formulation: str = None,
        retrieved_insights: List[dict] = [],
        diagnosed_issues: List[dict] = [],
        llm_opt: "ProgramGenerator" = None,
        llm_retri: "LibraryRetrieval" = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",            
    ) -> Tuple[bool, Optional[str]]:           
        """
        Diagnose the candidate mathematical formulation
        """
        pos_formulation_ins, program_ins = retrieved_insights

        # {issue_id: applicable_insights}, exclude pos_formulation_ins
        exclude_ids = [ins["insight_id"] for ins in pos_formulation_ins]
        issues_applicable_insights = llm_retri.retrieve_insights_for_diagnosis(
                iter=iter,
                task=task,
                formulation=failed_formulation,
                diagnosed_issues=diagnosed_issues,
                filter_fn=lambda ins: ins.insight_id not in exclude_ids, 
                verbose=verbose,
                save_data=save_data,
                output_path=output_path,
        )

        # Each issue may return duplicated insights; dedup per-issue first.
        candidate_ins_set = [
            self._dedup_inner_list_by_id(insights)
            for insights in issues_applicable_insights.values()
            if insights  # * remove empty insight list for any issue
        ]

        is_generate_new = True
        # all_solved = False
        combo_issues_status = []
        combo_unretrieved_ins = []
        combo_corrected_forms = []

        # Only iterate the subset combinations
        max_combo_size = 8
        # Use islice to cap the number of UNIQUE combinations (dedup within combo and across combos by insight_id).
        for idx, unretrieved_ins in enumerate(
            itertools.islice(self.unique_combo_gen(candidate_ins_set), max_combo_size)
        ):
            combo_unretrieved_ins.append(unretrieved_ins)

            formulation_ins = pos_formulation_ins + unretrieved_ins
            # Generate new formulation
            corrected_formulation = llm_opt.generate_formulation(
                iter=iter,
                task=task,
                retrieved_insights=formulation_ins,
                # rewrite=False,
                abl_params=config.ablation,
                verbose=False,
                save_data=True,
                output_path=output_path
            )
            combo_corrected_forms.append(corrected_formulation)

            # Construct the prompt for diagnosis
            prompt = PROMPT_VALIDATE_ISSUES.format(
                problem_description=task.desc,
                failed_formulation=failed_formulation,
                diagnosed_issues=json.dumps(diagnosed_issues),
                new_formulation=corrected_formulation
            )

            # Call the LLM to generate the answer and extract code from string 
            log_header = (f"\n==========\n[Iteration {iter}] Validate the regenerated mathematical formulation based on NO.{idx+1} unretrieved insights set for Task {task.id}\n==========\n")
            error_message = f"\n   Task {task.id} failed to validate regenerated mathematical formulation from LLM after maximum attempts\n"
            try:
                issues_status = call_llm_and_parse_with_retry(
                    model       = self.model,
                    service     = self.service,
                    prompt      = prompt,
                    # Extract code script from LLM response
                    parse_fn    = extract_json_array,
                    temperature = self.temp,
                    max_retry   = 5,                  
                    sleep_sec   = 2,
                    verbose     = verbose,
                    log_header  = log_header,
                    error_message = error_message
                )
                
            except Exception as err:
                print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to validate regenerated mathematical formulation\n")
                traceback.print_exc() # print error and cause
                issues_status = []

            # Save the issues status for each combination
            combo_issues_status.append(issues_status)

            #* If all issues are solved, try to generate program and check optimality 
            all_solved = all(item["status"] == "solved" for item in issues_status)
            if all_solved:
                # {insight_id: unretrieved}
                insights_diag = {item["insight_id"]: "unretrieved" for item in unretrieved_ins}
                _, output, _, _ = llm_opt.generate_program(
                    iter=iter,
                    task=task,
                    retrieved_insights=program_ins,
                    formulation=corrected_formulation,
                    abl_params=config.ablation,
                    verbose=False,
                    save_data=True,
                    output_path=output_path
                )

                # Check optimality
                if isinstance(output, (float, int)) and is_optimal_with_tolerance(output=output, gt=task.ground_truth):
                    # It is not necessary to generate new insights
                    is_generate_new = False 
                    new_formulation = None
                    break
                # All solved but not optimal
                else:
                    # is_generate_new = True
                    new_formulation = corrected_formulation
                    break
        
        if (not all_solved) and is_generate_new and combo_issues_status:
            # Count the number of "unsolved" items within each combo
            unsolved_counts = [sum(1 for issue in combo if issue["status"] == "unsolved") for combo in combo_issues_status]
            if unsolved_counts:
                min_count = min(unsolved_counts)
                target_idx = unsolved_counts.index(min_count)  # Choose combination with least unsolved issues
            else:
                target_idx = 0
            
            unretrieved_ins = combo_unretrieved_ins[target_idx]
            new_formulation = combo_corrected_forms[target_idx]
            issues_status = combo_issues_status[target_idx]

            # insights_diag = {
            #     unretrieved_ins[idx]["insight_id"]: "unretrieved"
            #     for idx, issue in enumerate(issues_status)
            #     if issue["status"] == "solved"
            # }

            insights_diag = {
                ins["insight_id"]: "unretrieved"
                for ins, st in zip(unretrieved_ins, issues_status)
                if isinstance(st, dict) and st.get("status") == "solved"
            }

        if save_data:
            formu_path = f"{output_path}/Diagnosis/formu2_iter_{iter}.py"
            issues_path = f"{output_path}/Diagnosis/issues_status_iter_{iter}.json"
            save_log_data(new_formulation, formu_path)
            save_log_data(issues_status, issues_path)
            
        return insights_diag, unretrieved_ins, is_generate_new, new_formulation


    def diagnose_formulation(
        self,
        iter: int = None,
        task: "Task" = None,
        feedback: str = None,
        failed_formulation: str = None,
        retrieved_insights: List[dict] = [],
        llm_opt: "ProgramGenerator" = None,
        llm_retri: "LibraryRetrieval" = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ) -> Tuple[bool, Optional[str]]:           
        """
        Diagnose failed formulation and the effectiveness of retrieved insights
        """

        #* Step 1: Diagnose the issues in the failed formulation
        diagnosed_issues = self._diagnose_issues(
            iter=iter,
            task=task,
            failed_formulation=failed_formulation,
            feedback=feedback,
            verbose=False, #verbose,
            save_data=save_data,
            output_path=output_path
        )

        #* Step 2: Diagnose the state of retrieved insights (positive or negative)
        insights_diag, pos_formulation_ins, is_retrieve_new, corrected_formulation, updated_issues = self._diagnose_pos_neg(
            iter=iter,
            task=task,
            failed_formulation=failed_formulation,
            diagnosed_issues=diagnosed_issues,
            retrieved_insights=retrieved_insights,
            llm_opt=llm_opt,
            verbose=False,
            save_data=save_data,
            output_path=output_path
        )

        if not is_retrieve_new:
            print("The retrieved insights are sufficient to solve the task after removing negative insights!")
            is_generate_new = False
            return insights_diag, is_generate_new, None, None 

        else:
            retrieved_insights[0] = pos_formulation_ins # Only keep positive insights
            #* Step 3: Diagnose the unretrieved insights
            insights_diag_new, unretrieved_ins, is_generate_new, new_formulation = self._diagnose_unretrieved(
                iter=iter,
                task=task,
                failed_formulation=corrected_formulation,
                retrieved_insights=retrieved_insights,
                diagnosed_issues=updated_issues,
                llm_opt=llm_opt,
                llm_retri=llm_retri,
                verbose=False,
                save_data=save_data,
                output_path=output_path,            
            )  

            if not is_generate_new:
                print("The existing insights are sufficient to solve the task after adding unretrieved insights!")

            insights_diag.update(insights_diag_new)
            
            updated_formulation_ins = pos_formulation_ins + unretrieved_ins

            return insights_diag, updated_formulation_ins, is_generate_new, new_formulation


    def debug_program(
        self,
        iter: int = None,
        task: "Task" = None,
        failed_program: str = None,
        feedback: str = None,
        verbose: bool = False,
        save_data: bool = False,
        output_path: str = "learning",
    ) -> Tuple[bool, Optional[str]]:           
        """
        Diagnose and correct the failed program with LLM
        """
        max_retry_correct = 8
        runnable = False                    
        current_program  = failed_program
        current_feedback = feedback

        for attempt in range(1, max_retry_correct + 1):

            # Construct the prompt for diagnosis
            prompt = PROMPT_PROGRAM_DIAG.format(
                failed_program      = current_program,
                feedback            = current_feedback,
                # correct_program     = task.correct_program,
            )

            # Call the LLM to generate the answer and extract code from string 
            log_header = (f"\n==========\n[Iteration {iter}] Diagnose and correct the failed program for Task {task.id} at attempt {attempt} \n==========\n")
            error_message = f"\n   Task {task.id} failed to extract code from LLM after maximum attempts\n"
            
            try:
                corrected_program = call_llm_and_parse_with_retry(
                    model       = self.model,
                    service     = self.service,
                    prompt      = prompt,
                    # Extract code script from LLM response
                    parse_fn    = self.extract_code,
                    temperature = self.temp,
                    max_retry   = 5,                  
                    sleep_sec   = 2,
                    verbose     = verbose,
                    log_header  = log_header,
                    error_message = error_message
                )

                # Update prompt context with new failed program
                current_program  = corrected_program

            except Exception as err:
                print(f"\n   [WARNING] Task {task.id}: Handle malformed LLM outputs after maximum retry as failing to correct program\n")
                traceback.print_exc() # print error and cause
                return False, None, None, None

            # Save and run corrected code 
            program_path = (f"{output_path}/corrected_program_iter_{iter}.py")
            save_log_data(corrected_program, program_path)

            #* Execute the corrected program
            try:
                output = self.execute_code(program_path) 
                runnable = True
                is_time_out = False
                #* Add solver time limitation to avoid large time cost on solving single task
                if isinstance(output, subprocess.TimeoutExpired):
                    print(f"\n   [Task {task.id}] exceeded maximum run time and was terminated\n")
                    is_time_out = True
                else:
                    try:
                        output = float(output) # ensure numerical outputs

                    except (TypeError, ValueError):
                        pass # keep original output

                # Check optimality when the program is runnable
                is_optimal, output_status, current_feedback = check_optimality(task=task, output=output, runnable=runnable, is_time_out=is_time_out)
                # if runnable, output the current feedback
                return is_optimal, runnable, corrected_program, current_feedback

            except Exception as err:
                # Update prompt context with feedback about execution error
                current_feedback = f"Execution error:\n {err.stderr}"
                print(f"\n   [Task {task.id}] failed to execute program on attempt {attempt}:\n{err.stderr}.")

        # Reached maximum retry for correction without successful execution
        print(f"\n   [Task {task.id}]: Maximum retry reached. Failed to fix the program. Skip!")
        corrected_program = None
        is_optimal = None
        current_feedback = None

        return is_optimal, runnable, corrected_program, current_feedback


# Test on a demo
if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from experience_library import ExperienceLibrary