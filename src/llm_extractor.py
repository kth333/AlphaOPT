import os
import re
import json
import copy
import subprocess
from itertools import chain
from typing import Optional, List, Tuple
import traceback

from .utils import save_log_data, call_llm_and_parse_with_retry, extract_json_array, extract_json_object
from .dataloader import DataLoader, Task
from .prompts.prompts_ins import PROMPT_INS_FROM_FORMU, PROMPT_INS_FROM_PROGRAM, PROMPT_CONDUCT_MERGE, PROMPT_ONLINE_MERGE

class InsightExtractor:
    """
    LLM_ins agent: Extract labeled insights from code corrections
    """
    def __init__(self, model: str, service: str, temperature: float | None = None):
        self.model = model
        self.service = service
        self.temp = temperature


    def extract_insights(self, text: str) -> dict:
        candidate = None
        try:
            raw = text
            # Extract content between the first '{' and the matching last '}'
            s, e = raw.find('['), raw.rfind(']')
            if s != -1 and e != -1 and e > s:
                candidate = raw[s:e+1]
            else:
                # Grab outermost '{' ... '}' (single object) and wrap later
                s, e = raw.find('{'), raw.rfind('}')
                if s == -1 or e == -1 or e <= s:
                    raise ValueError("No JSON array/object found in LLM output.")
                candidate = raw[s:e+1]

            cand = candidate.strip()

            # Remove trailing commas before ']' or '}'
            cand = re.sub(r",\s*(\]|\})", r"\1", cand)

            # Escape invalid backslashes (i.e., not followed by a valid JSON escape char)
            cand = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cand)

            # Parse JSON
            result = json.loads(cand)
            # Normalize to list
            if isinstance(result, dict):
                result = [result]
            if not isinstance(result, list):
                raise ValueError(f"Parsed content is not a list; got {type(result).__name__}")
            # Ensure every element is a dict
            for idx, item in enumerate(result):
                if not isinstance(item, dict):
                    raise ValueError(f"Insight at index {idx} is not a dict: {type(item).__name__}")
            
            return result

        except Exception as e:
            # Diagnostic output
            print("LLM raw text:\n", text)
            print("Extracted JSON candidate:\n", candidate if candidate is not None else '<no candidate>')
            print("Error during extract_insights:", repr(e))
            raise


    def generate_insights(
        self, 
        iter: int = None, 
        task: "Task" = None, 
        corrected_program: str = None, 
        failed_formulation: str = None, 
        taxonomy: List[dict] = None,
        candidate_formulation: Optional[str] = None,
        verbose: bool = True,
        save_data: bool = False,
        output_path: str = "learning"
        ):
        
        if failed_formulation:
            stage = "Formulation"
            # Extract new insights based on comparison between proposed formulation and gold-standard program
            prompt = PROMPT_INS_FROM_FORMU.format(
                            problem_description=task.desc, 
                            failed_formulation=failed_formulation,
                            correct_program=task.correct_program,
                            domain_taxo=json.dumps(taxonomy["Domain Modeling"], indent=2, ensure_ascii=False),
                            formulation_taxo=json.dumps(taxonomy["General Formulation"], indent=2, ensure_ascii=False),
                            )
        elif corrected_program:
            stage = "Program"
            # Extract new insights based on fixed program
            prompt = PROMPT_INS_FROM_PROGRAM.format(
                            corrected_program=corrected_program,
                            candidate_formulation=candidate_formulation if candidate_formulation else "",
                            code_taxo=json.dumps(taxonomy["Code Implementation"], indent=2, ensure_ascii=False)
                            )

        custom_header = f"\n==========\n[Iteration {iter}] Generate insights for Task {task.id}\n==========\n"
        error_message = f"\n   Task {task.id} failed to extract generated insights after maximum attempts\n"

        try:
            # Call the LLM and parse the output
            new_insights = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                # Extract insights from LLM response
                parse_fn=self.extract_insights, 
                temperature=self.temp,
                max_retry=3,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message,
            )

        except Exception as err:
            print(f"\n   [WARNING] Task {task.id} Handle malformed LLM outputs after maximum retry as no insight generated\n")
            traceback.print_exc() # print error and cause
            return []

        # Enrich each insight with default id and task metadata
        for i, ins in enumerate(new_insights):
            new_insights[i] = {
                "insight_id": -1,        
                **ins,                      
                "task_id": task.id,
                "iteration": iter        
            }
        
        if save_data:
            # Save the insights to a JSON file
            insights_path = f"{output_path}/{stage}/extracted_insights_iter_{iter}.json"
            new_insights_copy = copy.deepcopy(new_insights)
            for ins in new_insights_copy:
                taxo = ins.get("taxonomy", {})
                original_taxo = copy.deepcopy(taxo)  # Save original for debugging

                # ====== Repair taxonomy using canonical mapping when level-1 is missing ====== #
                # This mirrors the logic in ExperienceLibrary.add_insights.
                def _load_canonical_level2_to_level1() -> dict:
                    canonical_path = "./data/experience_library/iterations/train_data_4o_flash/latest_taxonomy_refine_iter1.json"
                    if not os.path.isfile(canonical_path):
                        return {}

                    try:
                        with open(canonical_path, "r", encoding="utf-8") as f:
                            canonical_taxo = json.load(f)
                    except Exception:
                        return {}

                    mapping = {}
                    # canonical structure: {stage: {level1: {level2: ...}}}
                    for stage, lvl1_map in canonical_taxo.items():
                        if not isinstance(lvl1_map, dict):
                            continue
                        for lvl1, lvl2_map in lvl1_map.items():
                            if not isinstance(lvl2_map, dict):
                                continue
                            for lvl2 in lvl2_map.keys():
                                mapping.setdefault(str(lvl2), (stage, lvl1))
                    return mapping

                canonical_level2_to_level1 = _load_canonical_level2_to_level1()

                if isinstance(taxo, dict) and canonical_level2_to_level1:
                    repaired_taxo = copy.deepcopy(taxo)
                    for stage_k, lvl1_val in list(taxo.items()):
                        if not isinstance(lvl1_val, dict):
                            continue
                        for maybe_lvl1, maybe_lvl2_val in list(lvl1_val.items()):
                            # Value is not a dict -> likely missing level-1; key is actually level-2
                            if not isinstance(maybe_lvl2_val, dict):
                                lvl2_label = str(maybe_lvl1)
                                if lvl2_label in canonical_level2_to_level1:
                                    canon_stage, canon_lvl1 = canonical_level2_to_level1[lvl2_label]
                                    repaired_taxo.setdefault(canon_stage, {})
                                    if not isinstance(repaired_taxo[canon_stage], dict):
                                        repaired_taxo[canon_stage] = {}
                                    repaired_taxo[canon_stage].setdefault(canon_lvl1, {})
                                    if not isinstance(repaired_taxo[canon_stage][canon_lvl1], dict):
                                        repaired_taxo[canon_stage][canon_lvl1] = {}
                                    repaired_taxo[canon_stage][canon_lvl1][lvl2_label] = None
                                    try:
                                        del repaired_taxo[stage_k][maybe_lvl1]
                                        if not repaired_taxo[stage_k]:
                                            del repaired_taxo[stage_k]
                                    except Exception:
                                        pass
                    taxo = repaired_taxo

                norm = {}
                # lvl1 = stage (e.g., "General Formulation"); lvl1_val = {level1_name: {level2: (null|str)}}
                for lvl1, lvl1_val in taxo.items():
                    if not isinstance(lvl1_val, dict):
                        norm[lvl1] = {}
                        continue
                    # Map each level1_name to the list of all level2 label names under it.
                    norm[lvl1] = {
                        level1_name: (list(level2_dict.keys()) if isinstance(level2_dict, dict) else [])
                        for level1_name, level2_dict in lvl1_val.items()
                    }
                ins["taxonomy"] = norm
                
                # Debug: Check for empty arrays in normalized taxonomy
                has_empty_array = False
                for lvl1, lvl1_val in norm.items():
                    if isinstance(lvl1_val, dict):
                        for lvl2_name, lvl2_list in lvl1_val.items():
                            if isinstance(lvl2_list, list) and len(lvl2_list) == 0:
                                has_empty_array = True
                                break
                    if has_empty_array:
                        break
                
                # Debug: Check for ["definition", "condition"] pattern in original taxonomy
                has_definition_condition_list = False
                for lvl1, lvl1_val in original_taxo.items():
                    if isinstance(lvl1_val, dict):
                        for lvl2_name, lvl2_val in lvl1_val.items():
                            # Check if Level-2 value is a list containing "definition" and "condition"
                            if isinstance(lvl2_val, list) and "definition" in lvl2_val and "condition" in lvl2_val:
                                has_definition_condition_list = True
                                break
                            # Also check if Level-2 value is a dict with Level-3 values being ["definition", "condition"]
                            elif isinstance(lvl2_val, dict):
                                for lvl3_name, lvl3_val in lvl2_val.items():
                                    if isinstance(lvl3_val, list) and "definition" in lvl3_val and "condition" in lvl3_val:
                                        has_definition_condition_list = True
                                        break
                        if has_definition_condition_list:
                            break
                    if has_definition_condition_list:
                        break
                
                if has_empty_array:
                    print(f"\n[DEBUG] Empty taxonomy array detected in generate_insights (task_id: {task.id}, iter: {iter}, stage: {stage})")
                    print(f"[DEBUG] Original LLM taxonomy output:")
                    print(json.dumps(original_taxo, indent=2, ensure_ascii=False))
                    print(f"[DEBUG] Normalized taxonomy (with empty array):")
                    print(json.dumps(norm, indent=2, ensure_ascii=False))
                    print(f"[DEBUG] Full insight (excluding taxonomy):")
                    ins_debug = {k: v for k, v in ins.items() if k != "taxonomy"}
                    print(json.dumps(ins_debug, indent=2, ensure_ascii=False))
                    print("-" * 80)
                
                if has_definition_condition_list:
                    print(f"\n[DEBUG] Invalid taxonomy format detected: Level-3 contains ['definition', 'condition'] list")
                    print(f"[DEBUG] Task ID: {task.id}, Iteration: {iter}, Stage: {stage}")
                    print(f"[DEBUG] Original LLM taxonomy output:")
                    print(json.dumps(original_taxo, indent=2, ensure_ascii=False))
                    print(f"[DEBUG] Normalized taxonomy:")
                    print(json.dumps(norm, indent=2, ensure_ascii=False))
                    print(f"[DEBUG] Full insight (excluding taxonomy):")
                    ins_debug = {k: v for k, v in ins.items() if k != "taxonomy"}
                    print(json.dumps(ins_debug, indent=2, ensure_ascii=False))
                    print("-" * 80)
            
            save_log_data(new_insights_copy, insights_path)

        return new_insights


    def conduct_insight_merge(
        self, 
        candidate_insights: List[dict] = None, 
        target: int = None,
        verbose: bool = False
        ):
        mapping_ids = {ins["insight_id"]: ins["task_id"] for ins in candidate_insights}
        kept_fields = ["insight_id", "taxonomy", "condition", "explanation", "example"]
        insights_to_be_merge = [{k: d[k] for k in kept_fields if k in d} for d in candidate_insights]

        prompt = PROMPT_CONDUCT_MERGE.format(candidate_insights=json.dumps(insights_to_be_merge, indent=2, ensure_ascii=False))

        custom_header = f"\n==========\nMerge insights in {target}\n==========\n"
        error_message = f"\n   {target} failed to conduct insight merge after maximum attempts\n"

        try:
            # Call the LLM and parse the output
            merge_results = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_array, 
                temperature=self.temp,
                max_retry=3,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                error_message=error_message,
            )

        except Exception as err:
            print(f"\n   [WARNING] {target} Handle malformed LLM outputs after maximum retry as no insight merge\n")
            traceback.print_exc() # print error and cause
            return []
        
        
        # Remove 'reason' field ("insight_id": -1)
        merged_insights = [
            {
                **{k: v for k, v in candidate.items() if k != "reason"},
                # flatten the mapping_ids[mid] list and deduplicate
                "task_id": list(set(chain.from_iterable(
                    [mapping_ids[mid]] if isinstance(mapping_ids[mid], str) else mapping_ids[mid]
                    for mid in candidate.get("merged_ids", [])
                    if mid in mapping_ids
                )))
            }
            for candidate in merge_results
        ]

        return merged_insights


    def conduct_insight_online_merge(
        self, 
        new_insight: List[dict] = None, 
        library: "ExperienceLibrary" = None,
        verbose: bool = False
        ):
        
        # Retrieve the existing insights in the library that match the taxonomy of the new insight
        matched_taxo = new_insight[0].get("taxonomy", {})
        existing_insights = library.retrieve_by_taxonomy(query_taxonomy=matched_taxo, include_task_id=True)
        # If no existing insights match the taxonomy, return empty results
        if not existing_insights:
            # print("no existing insights match the taxonomy")
            return [], {}, []
        
        mapping_ids = {ins["insight_id"]: ins["task_id"] for ins in existing_insights}
        # Create mapping from task_id to iteration
        mapping_task_to_iter = {}
        for ins in existing_insights:
            task_id = ins["task_id"]
            iteration = ins["iteration"]
            if isinstance(task_id, list):
                # If task_id is a list, map each task_id to the iteration
                for tid in task_id:
                    mapping_task_to_iter[tid] = iteration
            else:
                # If task_id is a single value, map it directly
                mapping_task_to_iter[task_id] = iteration

        kept_fields = ["taxonomy", "condition", "explanation", "example"]
        new_insight_for_merge = [{k: d[k] for k in kept_fields if k in d} for d in new_insight]
        kept_fields.append("insight_id")
        existing_insights_for_merge = [{k: d[k] for k in kept_fields if k in d} for d in existing_insights]
        
        # Merge the new insight with the existing insights in the library
        prompt = PROMPT_ONLINE_MERGE.format(new_insight=json.dumps(new_insight_for_merge, indent=2, ensure_ascii=False),
                                            existing_insights=json.dumps(existing_insights_for_merge, indent=2, ensure_ascii=False))
        # print("prompt", prompt)
        try:
            # Call the LLM and parse the output
            merge_results = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_object, 
                temperature=self.temp,
                max_retry=3,
                sleep_sec=0.5,
                verbose=verbose,
            )

            # print("merge_results", merge_results)
        except Exception as err:
            print(f"\n   [WARNING] Online merge: Handle malformed LLM outputs after maximum retry as no insight merge\n")
            traceback.print_exc() # print error and cause
            return [], {}, []
        
        
        # Skip empty merge results (when LLM decides not to merge)
        if not merge_results or not merge_results.get("merged_ids"):
            return [], {}, []

        # Normalize merged parent ids (should be hashable insight_id values)
        merged_parent_ids = []
        for mid in (merge_results.get("merged_ids") or []):
            # Some malformed outputs might wrap ids as dicts; extract if possible.
            if isinstance(mid, dict):
                mid = mid.get("insight_id")
            # Cast numeric strings to int when possible (keep original otherwise).
            if isinstance(mid, str):
                s = mid.strip()
                if s.isdigit():
                    try:
                        mid = int(s)
                    except Exception:
                        pass
            merged_parent_ids.append(mid)
        # Drop Nones
        merged_parent_ids = [x for x in merged_parent_ids if x is not None]
        
        # Remove 'reason' field ("insight_id": -1) and create single merged insight
        new_task_id = new_insight[0]["task_id"]
        if isinstance(new_task_id, list):
            new_task_id_list = new_task_id
        else:
            new_task_id_list = [new_task_id]
        
        merged_insights = {
            **{k: v for k, v in merge_results.items() if k != "reason"},
            # Ensure merged_ids are normalized (hashable insight_id values)
            "merged_ids": merged_parent_ids,
            # flatten the mapping_ids[mid] list and deduplicate, then add new_insight task_id
            "task_id": list(set(list(chain.from_iterable(
                [mapping_ids[mid]] if isinstance(mapping_ids[mid], str) else mapping_ids[mid]
                for mid in merged_parent_ids
                if mid in mapping_ids
            )) + new_task_id_list))
        }
        # print("merged_insights", merged_insights)
        # Build iteration mapping table for all task_ids in merged_insights
        merged_task_to_iter = {}
        for task_id in merged_insights.get("task_id", []):
            if task_id in mapping_task_to_iter:
                merged_task_to_iter[task_id] = mapping_task_to_iter[task_id]
            # For new_insight's task_id, use the latest iteration
            elif task_id == new_insight[0]["task_id"]:
                merged_task_to_iter[task_id] = -1 

        # Return only the parent insight_ids that were actually merged (NOT the entire matched candidate list).
        return merged_insights, merged_task_to_iter, merged_parent_ids


    def modify_new_insight_for_retrieve(
        self,
        iter: int,
        task: "Task",
        insight: dict,
        taxonomy_failed: bool = False,
        condition_failed: bool = False,
        library: Optional["ExperienceLibrary"] = None,
        candidate_formulation: Optional[str] = None,
        verbose: bool = True
    ) -> dict:
        """
        Modify a newly generated insight to improve its retrievability for the given task.
        - If taxonomy_failed is True: adjust the taxonomy so that it better matches the task.
        - If condition_failed is True: adjust the applicability condition.
        """

        modified_insight = copy.deepcopy(insight)
        simplified_insight = {
            "taxonomy": insight.get("taxonomy", {}),
            "condition": insight.get("condition", ""),
            "explanation": insight.get("explanation", "")
        }
        modified_insight_str = json.dumps(simplified_insight, ensure_ascii=False, indent=2)
        taxonomy_dict = json.dumps(library.taxonomy, indent=2, ensure_ascii=False)

        # 1) Modify taxonomy if needed
        if taxonomy_failed:
            if "Code Implementation" in simplified_insight.get("taxonomy", {}):
                # Use candidate_formulation if provided, otherwise fallback to task.math_model
                problem_desc_or_math_model = candidate_formulation
                taxonomy_rewrite_rule = """
                    Your task is to rewrite the "taxonomy" field of an insight so that it can be better retrieved when only reading the given mathematical model of the optimization task. 
                    When matching the insight to the mathematical model, except the label name, the two fields are important:
                        - "definition" — what the label means (scope/intent).
                        - "condition" — when to apply the label (a general trigger explicitly grounded in the mathematical model).
                """
            else:
                problem_desc_or_math_model = task.desc
                taxonomy_rewrite_rule = """
                    Your task is to rewrite the "taxonomy" field of an insight so that it can be better retrieved when only reading the given problem description of the optimization task. 
                    When matching the insight to the problem description, except the label name, the two fields are important:
                        - "definition" — what the label means (scope/intent).
                        - "condition" — when to apply the label (a general trigger explicitly grounded in the problem description or in the defining features of the problem domain).
                """

            prompt = """
                You are an expert in Industrial Engineering and Operations Research.

                You are given:

                ### Problem description or Mathematical model of the optimization task
                {problem_desc_or_math_model}

                ### Original insight
                {modified_insight_str}

                ### Current Taxonomy Dictionaries
                {taxonomy_dict}

                ### Your Task
                {taxonomy_rewrite_rule}

                You have three options to modify the taxonomy:

                **Option a) Modify the definition and condition of the current taxonomy label:**
                - Keep the same track, level-1, and level-2 labels as in the original insight
                - Only modify the "definition" and "condition" fields of the current level-2 label
                - **Retain the original phrasing whenever possible and introduce changes additively**, so as to minimize disruption to existing task-label matching.

                **Option b) Replace the current taxonomy label with an existing one from the taxonomy dictionary:**
                - Select an existing level-1 and level-2 label from the taxonomy dictionary provided above
                - You may optionally modify the existing level-2 label's "definition" and "condition" to better fit this insight
                - If you do not modify them, set both "definition" and "condition" to `null`
                - Do not keep the original labels, replace them with the new ones

                **Option c) Invent a new taxonomy label:**
                - Only use this option if no suitable existing label is available in the taxonomy dictionary
                - Create a new level-2 label (or level-1 if necessary) with appropriate "definition" and "condition"
                - Ensure the new label is general and reusable, not overly specific to this single problem

                ### Strict Output Format

                Return a single JSON object of the form:
                {{
                "taxonomy": {{
                    "Domain Modeling" | "General Formulation" | "Code Implementation" : {{
                    <Level-1 label> : {{
                        <Level-2 label> : null | {{ "definition": "...", "condition": "..." }}
                    }}
                  }}
                }}
                }}

                **Important Note**: 
                1 When modifying the taxonomy, you should **first try to select an existing taxonomy label from the dictionaries above**. Only invent a new taxonomy label if no suitable existing one is available.
                2 The taxonomy of an insight may be switched between 'Domain Modeling' and 'General Formulation', but must not be reassigned to or from 'Code Implementation'.
                """

            prompt = prompt.format(problem_desc_or_math_model=problem_desc_or_math_model, modified_insight_str=modified_insight_str, taxonomy_dict=taxonomy_dict, taxonomy_rewrite_rule=taxonomy_rewrite_rule)

            try:
                resp = call_llm_and_parse_with_retry(
                    model=self.model,
                    service=self.service,
                    prompt=prompt,
                    parse_fn=extract_json_object,
                    temperature=self.temp,
                    max_retry=3,
                    sleep_sec=0.5,
                    verbose=verbose,
                    log_header=f"\n==========\n[Iteration {iter}] Modify taxonomy for retrieval - Task {task.id}\n==========\n",
                    error_message=f"\n   Task {task.id} failed to modify taxonomy for retrieval\n",
                )
                print("⭐️ modify_new_insight_for_retrieve - taxonomy", resp)
                new_taxo = resp.get("taxonomy")

                if isinstance(new_taxo, dict):
                    # Convert taxonomy from object format to array format
                    # From: {"Domain Modeling": {"Level-1": {"Level-2": null | {...}}}}
                    # To:   {"Domain Modeling": {"Level-1": ["Level-2"]}}
                    converted_taxo = {}
                    for track, lvl1_map in new_taxo.items():
                        if not isinstance(lvl1_map, dict):
                            continue
                        converted_taxo[track] = {}
                        for lvl1, lvl2_map in lvl1_map.items():
                            if isinstance(lvl2_map, dict):
                                # Extract all Level-2 label keys into an array
                                lvl2_labels = list(lvl2_map.keys())
                                converted_taxo[track][lvl1] = lvl2_labels
                            else:
                                # If Level-1 value is not a dict, keep as is (shouldn't happen normally)
                                converted_taxo[track][lvl1] = lvl2_map
                    modified_insight["taxonomy"] = converted_taxo
            except Exception as err:
                print(f"[WARNING] Task {task.id}: modify_new_insight_for_retrieve (taxonomy) failed with error: {err}")
                traceback.print_exc()

        # 2) Modify condition if needed
        if condition_failed:
            if "Code Implementation" in simplified_insight.get("taxonomy", {}):
                # Use candidate_formulation if provided, otherwise fallback to task.math_model
                problem_desc_or_math_model = candidate_formulation
                condition_rewrite_rule = """
                    Your task is to rewrite the "condition" field of an insight so that it can be better matched when reading the given mathematical model of the optimization task. 

                    Write it as a trigger explicitly grounded in the mathematical model. State the general modeling pattern, then use this model as an example. **Use the pattern**: "This insight applies when the mathematical model contains... For example, when the formulation included...". Keep it strictly non-prescriptive: do not give any solution, advice or decision.
                """
            else:
                problem_desc_or_math_model = task.desc
                condition_rewrite_rule = """
                    Your task is to rewrite the "condition" field of an insight so that it can be better matched when reading the given problem description of the optimization task. 

                    Write the condition as a trigger explicitly grounded in the problem description or in the defining features of the problem domain. First state the general situation, then use this problem as an example. **Use the pattern**: "This insight applies when ... For example, when the problem statement mentioned ...". Keep it strictly non-prescriptive: do not give any solution, advice or decision.
                """
            prompt = """
                You are an expert in Industrial Engineering and Operations Research.

                You are given:

                ### Problem description or Mathematical model of the optimization task
                {problem_desc_or_math_model}

                ### Original insight
                {modified_insight_str}

                ### Your Task

                {condition_rewrite_rule}

                ### Strict Output Format

                Return a JSON object of the form:
                {{
                "condition": "..."
                }}
                """

            prompt = prompt.format(problem_desc_or_math_model=problem_desc_or_math_model, modified_insight_str=modified_insight_str, condition_rewrite_rule=condition_rewrite_rule)

            try:
                resp = call_llm_and_parse_with_retry(
                    model=self.model,
                    service=self.service,
                    prompt=prompt,
                    parse_fn=extract_json_object,
                    temperature=self.temp,
                    max_retry=3,
                    sleep_sec=0.5,
                    verbose=verbose,
                    log_header=f"\n==========\n[Iteration {iter}] Modify condition for retrieval - Task {task.id}\n==========\n",
                    error_message=f"\n   Task {task.id} failed to modify condition for retrieval\n",
                )
                print("⭐️ modify_new_insight_for_retrieve - condition", resp)
                new_condition = resp.get("condition")
                modified_insight["condition"] = new_condition.strip()
            except Exception as err:
                print(f"[WARNING] Task {task.id}: modify_new_insight_for_retrieve (condition) failed with error: {err}")
                traceback.print_exc()

        return modified_insight


# Test on a demo
if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from experience_library import ExperienceLibrary

    iter = 1
    dataset="integrated_train_no_label"
    library = ExperienceLibrary()

    lock = Lock()               # Lock to safely update shared variables
    temp_lib = []

    def process_task(task, taxo_snapshot):

        # insights_path = f"./learning/{dataset}/task_{task.id}/applicable_insights_iter_{iter}.json"
        program_path  = f"./learning/{dataset}/task_{task.id}/corrected_program_iter_{iter}.py"

        if os.path.exists(program_path):
            with open(program_path, "r", encoding="utf-8") as f:
                corrected_program = f.read()

            output_path = f"./learning/{dataset}/task_{task.id}/labeled_ins/fulltaxo/"
            os.makedirs(output_path, exist_ok=True)

            llm_ins = InsightExtractor(model="gemini-2.5-pro")
            new_insights = llm_ins.generate_insights(
                iter=iter,
                task=task,
                corrected_program=corrected_program,
                taxonomy=taxo_snapshot,
                verbose=True,
                save_data=True,
                output_path=output_path
            )

            return new_insights

    train_dataset_path = f"./learning/{dataset}/train_tasks_record_iter{iter}.json"
    tasks = DataLoader(train_dataset_path, mode="learn", filter_success_num=None, reset=False)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_task, task, copy.deepcopy(library.taxonomy)) #* Pass a taxonomy snapshot to each task (to avoid concurrent writes)
            for task in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks\n"):
            new_insights = future.result()
            if new_insights:    
                #* Temporarily store new insights in each iteration
                with lock:
                    temp_lib.extend(new_insights)
                #* Update the shared taxonomy snapshot                           
                library.update_taxonomy(new_insights)


    #* Add the new insights into the experience library
    library.add_insights(temp_lib)
    library.save(f"./data/experience_library/iterations/integrated_train_new_label/library_iter{iter}_fulltaxo.json")
    # Save updated taxonomy
    library.save_taxonomy(f"./data/experience_library/iterations/integrated_train_new_label/latest_taxonomy_iter{iter}_fulltaxo.json")
