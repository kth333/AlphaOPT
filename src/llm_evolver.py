from __future__ import annotations # Enable postponed (lazy) evaluation of type hints
import os
import re
import glob
import json
import numpy as np
from typing import List, Any
import copy
from tqdm import tqdm
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed

from .experience_library import ExperienceLibrary
from .utils import save_log_data, extract_json_object, extract_json_array, call_llm_and_parse_with_retry
from .dataloader import DataLoader 
from .llm_retriever import LibraryRetrieval
from .prompts.prompts_evolve import PROMPT_INS_NEG, PROMPT_INS_UNR, PROMPT_INS_REFINEMENT

#* Configure
from omegaconf import OmegaConf
config = OmegaConf.load("train_config.yaml")

class LibraryEvolution:
    """
    Library evolution with LLM agents (currently work as insights merge)
    """
    def __init__(self, lib: "ExperienceLibrary", model: str, service: str, temperature: float | None = None):
        self.library = lib     # Use an ExperienceLibrary instance
        # self.client = client
        self.service = service
        self.model = model
        self.temp = temperature

    def verify_retrieval(
        self,
        ins_id,
        tasks,
        task_ids,
        llm_retri,
        *,
        stage: str = "Formulation",
        config_override=None,
        formulation_lookup=None,
    ):
        count = 0
        tasks_subset = tasks.subset_by_ids(task_ids)
        matched_task_ids = []
        for task in tasks_subset:
            formulation = None
            if stage == "Program" and formulation_lookup is not None:
                try:
                    formulation = formulation_lookup(task)
                except Exception:
                    # Best-effort: proceed without formulation if lookup fails.
                    formulation = None

            formulation_ins = llm_retri.retrieve_applicable_insights(
                    iter=None,
                    task=task,
                    stage=stage,
                    formulation=formulation,
                    config=(config_override or config),
                    verbose=False,
                    save_data=False
                )
            retrieved_ins_ids = [ins["insight_id"] for ins in formulation_ins if 'insight_id' in ins]
            if ins_id in retrieved_ins_ids:
                count += 1
                matched_task_ids.append(task.id)

        return count, matched_task_ids


    def build_library_variant(self, ins_id, refined_conditions_k):
        lib_variants = []
        for ref_condition in refined_conditions_k:
            lib_variant = copy.deepcopy(self.library)
            for ins in lib_variant:
                if ins.insight_id == ins_id:
                    ins.condition = ref_condition
                    break
            lib_variants.append(lib_variant)
        return lib_variants


    def generate_neg_condition(self, task, insight, iter, verbose=False, output_dir=""):
        neg_evidence = ""
        
        try:
            # If output_dir is empty/None, do not attempt filesystem reads.
            # This avoids constructing absolute paths like "/task_xxx/..." that always fail.
            if output_dir:
                diag_path = os.path.join(
                    str(output_dir),
                    f"task_{task.id}",
                    "Diagnosis",
                    f"ins_pos_neg_diagnosis_iter_{iter}.json",
                )
                if os.path.isfile(diag_path):
                    with open(diag_path, encoding="utf-8") as f:
                        pos_neg_ins_diag = json.load(f)
                    neg_evidence = next(
                        (
                            d.get("evidence")
                            for d in pos_neg_ins_diag
                            if d.get("insight_id") == insight.insight_id
                        ),
                        "",
                    )

        except Exception:
            # Best-effort evidence; proceed with empty diag_evidence.
            if verbose:
                print(f"[WARN] generate_neg_condition: missing/invalid diagnosis evidence for task {task.id}")

        # target_ins = {"condition": insight.condition, "explanation": insight.explanation, "example": insight.example}
        target_ins = {"condition": insight.condition, "explanation": insight.explanation}
        prompt = PROMPT_INS_NEG.format(
            target_insight=json.dumps(target_ins),
            desc=task.desc,
            diag_evidence=neg_evidence
        )    
        # print(prompt)

        neg_condition = []
        try:
            error_message = f"\n   insight {insight.insight_id} failed to be refined for {task.id} from LLM after maximum attempts\n"
            # Call the LLM and parse the output
            refined_result = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_object,
                temperature=self.temp,
                max_retry=3,
                sleep_sec=0.5,
                verbose=verbose,
                error_message=error_message 
            )
            # print(refined_result)

            if refined_result: # If output {} which means the insight is applicable
                neg_condition = refined_result["condition"]

        except Exception as err:
            print(f"\n   [WARNING]: Handle malformed LLM outputs after maximum retry as no refinement for insight {insight.insight_id} on {task.id}\n")
            traceback.print_exc() # print error and cause

        return neg_condition


    def generate_unr_condition(self, task, insight, iter, verbose=False, output_dir=""):
        # Retrieve insight diagnosis reason
        unr_evidence = ""
        try:
            # If output_dir is empty/None, do not attempt filesystem reads.
            if output_dir:
                pattern = os.path.join(
                    str(output_dir),
                    f"task_{task.id}",
                    "Diagnosis",
                    f"applicable_insights_iter_{iter}_idx*.json",
                )
                files = glob.glob(pattern)
                for fp in sorted(files):
                    with open(fp, "r", encoding="utf-8") as f:
                        unr_ins_diag = json.load(f)
                        unr_ins_diag = unr_ins_diag.get("applicable_insights")
                    for diag in unr_ins_diag or []:
                        if diag.get("insight_id") == insight.insight_id:
                            unr_evidence = diag.get("reason") or diag.get("evidence")
                            break
                    if unr_evidence:
                        break
        except Exception:
            # Best-effort evidence; proceed with empty diag_evidence.
            if verbose:
                print(f"[WARN] generate_unr_condition: missing/invalid diagnosis evidence for task {task.id}")

        
        # target_ins = {"condition": insight.condition, "explanation": insight.explanation, "example": insight.example}
        target_ins = {"condition": insight.condition, "explanation": insight.explanation}
        prompt = PROMPT_INS_UNR.format(
            target_insight=json.dumps(target_ins),
            desc=task.desc,
            diag_evidence=unr_evidence
        )    

        unr_condition = []
        try:
            error_message = f"\n   insight {insight.insight_id} failed to be refined for {task.id} from LLM after maximum attempts\n"
            # Call the LLM and parse the output
            refined_result = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_object,
                temperature=self.temp,
                max_retry=3,
                sleep_sec=0.5,
                verbose=verbose,
                error_message=error_message 
            )
            # print(refined_result)

            if refined_result:
                unr_condition = refined_result["condition"]

        except Exception as err:
            print(f"\n   [WARNING]: Handle malformed LLM outputs after maximum retry as no refinement for insight {insight.insight_id} on {task.id}\n")
            traceback.print_exc() # print error and cause

        return unr_condition

    def refine_insight(self, iter, neg_condition_lst, unr_condition_lst, insight, path_k=5, verbose=False):
        # Format insights for the prompt
        # target_ins = {k: insight.get(k) for k in ("condition", "explanation", "example")}
        # target_ins = {"condition": insight.condition, "explanation": insight.explanation, "example": insight.example}

        # Format numbered lists for the prompt
        neg_conditions_str = "\n".join(f"{i}. {cond}" for i, cond in enumerate(neg_condition_lst, start=1)) or ""
        unr_conditions_str = "\n".join(f"{i}. {cond}" for i, cond in enumerate(unr_condition_lst, start=1)) or ""

        # Build refinement prompt - only pass the condition, not the entire insight
        prompt = PROMPT_INS_REFINEMENT.format(
            # original_insight=json.dumps(target_ins),
            original_condition=insight.condition,
            neg_conditions=neg_conditions_str,
            unr_conditions=unr_conditions_str,
            path_k=path_k
        )

        # Call LLM for refined conditions (K variants)
        try:
            custom_header = (f"\n==========\n[Iteration {iter}] Refine insight condition {insight.insight_id}\n==========\n")
            error_message = f"\n   insight {insight.insight_id} failed to be refined from LLM after maximum attempts\n"
            refined_results = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_array,
                temperature=self.temp,
                max_retry=3,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message
            )
        except Exception:
            traceback.print_exc()
            refined_results = []

        refined_conditions_k = [res["new_condition"] for res in refined_results] if refined_results else []

        return refined_conditions_k