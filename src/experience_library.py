import os
import json
from typing import List, Dict, Any, Optional, Callable
from copy import deepcopy
import itertools

def _coerce_taxonomy(taxo: Any) -> dict:
    """
    Coerce various taxonomy formats into the expected dict shape.

    Expected taxonomy shape (high-level):
      { "<Stage>": { "<Level1>": ["<Level2>", ...] } }
    But we defensively accept:
      - None -> {}
      - a JSON string -> parsed value (if dict)
      - a stage string like "General Formulation" -> {"General Formulation": {}}
      - any non-dict -> {}
    """
    if taxo is None:
        return {}
    if isinstance(taxo, dict):
        return taxo
    if isinstance(taxo, str):
        s = taxo.strip()
        # Try parsing JSON if it looks like an object
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                # Fall through to stage-string coercion
                pass
        # Treat as invalid taxonomy (do not invent structure here).
        return {}
    return {}


def _parse_taxonomy_strict(taxo: Any) -> Optional[dict]:
    """
    Strict taxonomy validator for NEW insights:
    - Accept dict as-is
    - Accept JSON string that parses to dict
    - Reject plain strings like "General Formulation" (return None)
    - Reject other types (return None)
    """
    if isinstance(taxo, dict):
        return taxo
    if isinstance(taxo, str):
        s = taxo.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None
    return None


_STAGE_KEYS = ("Domain Modeling", "General Formulation", "Code Implementation")


def _taxonomy_has_any_label(taxo: Any) -> bool:
    """
    Return True iff taxonomy contains at least one usable (stage, level-1, level-2) triple.

    Accepted shapes:
      - Normalized: {stage: {level1: [level2, ...]}}
      - Raw:        {stage: {level1: {level2: ...}}}
    """
    if not isinstance(taxo, dict):
        return False

    for stage in _STAGE_KEYS:
        stage_map = taxo.get(stage)
        if not isinstance(stage_map, dict):
            continue

        for _, lvl2_val in stage_map.items():
            if isinstance(lvl2_val, list):
                if any(str(x).strip() for x in lvl2_val if x is not None):
                    return True
            elif isinstance(lvl2_val, dict):
                if any(str(k).strip() for k in lvl2_val.keys() if k is not None):
                    return True
            # other shapes (None/str/etc.) are treated as invalid for new insights

    return False


class Insight:
    def __init__(self, data: dict):
        self.insight_id = data.get("insight_id")
        # Some LLM/merge outputs may provide taxonomy as a string (e.g., "General Formulation").
        # Coerce it to a dict to avoid downstream crashes.
        self.taxonomy = _coerce_taxonomy(data.get("taxonomy"))
        self.condition = data.get("condition")
        self.explanation = data.get("explanation")
        self.example = data.get("example")
        self.task_id = data.get("task_id")
        self.iteration = data.get("iteration")

        # Version tracking
        self.merge_version = data.get("merge_version", 0)  # number of successful merges
        self.refine_version = data.get("refine_version", 0)  # number of successful refinements

        # Per-iteration counters (lists indexed by iteration number)
        # occurrence[i] = number of times retrieved in iteration i
        self.occurrence = data.get("occurrence", [])  # list of retrieve counts per iteration
        # correctness[i] = number of times led to success in iteration i
        self.correctness = data.get("correctness", [])  # list of success counts per iteration

        initial_dist = {"positive": [], "negative": [], "unretrieved": [], "irrelevant": [], "invalid": []}
        self.distribution = data.get("distribution", initial_dist) # how the insight work on target tasks

    def to_dict(self) -> dict:
        result = {
            "insight_id": self.insight_id,
            "taxonomy": self.taxonomy,
            "condition": self.condition,
            "explanation": self.explanation,
            "example": self.example,
            "iteration": self.iteration,
            "task_id": self.task_id,
            "merge_version": self.merge_version,
            "refine_version": self.refine_version,
            "occurrence": self.occurrence,    # list of retrieve counts per iteration
            "correctness": self.correctness,  # list of success counts per iteration
            "distribution": self.distribution
        }

        return result
    

class ExperienceLibrary:
    def __init__(self, insight_list: list | None = None):
        """
        Build an ExperienceLibrary from an empty library and a pre-defined taxonomy dictionary
        """
        self._library = []          # type: list[Insight]
        if insight_list:                 # Skip if None or []
            for ins in insight_list:
                self._library.append(Insight(ins))

        with open("./data/experience_library/fewshot_taxonomy.json", "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        # self._taxo_lock = threading.RLock()

    @classmethod
    def from_json_file(cls, library_path: str, taxonomy_path: Optional[str] = None) -> "ExperienceLibrary":
        """
        Read a JSON file (list of dicts) and return an ExperienceLibrary instance.
        """
        if not os.path.isfile(library_path):
            raise FileNotFoundError(library_path)
        with open(library_path, "r") as f:
            data = json.load(f)     # Data is a list[dict]

        inst = cls(insight_list=data)

        if taxonomy_path is not None:
            if not os.path.isfile(taxonomy_path):
                raise FileNotFoundError(taxonomy_path)
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                inst.taxonomy = json.load(f) 

        return inst
        
    def __getitem__(self, index: int) -> Insight:
        return self._library[index]

    def __setitem__(self, index: int, new_insight: Insight):
        self._library[index] = new_insight

    def __len__(self):
        return len(self._library)

    def to_json(self) -> list:
        return [ins.to_dict() for ins in self._library]

    def save(self, path: str):
        """
        Save the current library to a JSON file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def save_taxonomy(self, path: str):
        """
        Save the current taxonomy to a JSON file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.taxonomy, f, indent=2, ensure_ascii=False)
        
    def update_usage(self, insight_ids: list, success: bool):
        """
        Increment occurrence for all used insight_ids.
        If `success` is True, also increment correctness.
        DEPRECATED: Use update_retrieval_stats instead.
        """
        # insight_ids = [ins["insight_id"] for ins in update_lst]
        for ins in self._library:
            if ins.insight_id in insight_ids:
                ins.occurrence += 1
                if success:
                    ins.correctness += 1

    def update_retrieval_stats(self, insight_ids: list, iteration: int, success: bool = False):
        """
        Update occurrence and correctness statistics for retrieved insights in a given iteration.
        
        Args:
            insight_ids: List of insight IDs that were retrieved
            iteration: Current iteration number (0-indexed)
            success: Whether the retrieval led to success (optimal/positive)
        """
        for ins in self._library:
            if ins.insight_id in insight_ids:
                # Ensure occurrence and correctness lists are long enough
                while len(ins.occurrence) <= iteration:
                    ins.occurrence.append(0)
                while len(ins.correctness) <= iteration:
                    ins.correctness.append(0)
                
                # Increment occurrence for this iteration
                ins.occurrence[iteration] += 1
                
                # Increment correctness if successful
                if success:
                    ins.correctness[iteration] += 1

    def increment_merge_version(self, insight_id: int):
        """
        Increment merge_version for an insight after successful merge.
        
        Args:
            insight_id: The insight ID to update
        """
        for ins in self._library:
            if ins.insight_id == insight_id:
                ins.merge_version += 1
                return
        print(f"Warning: Insight {insight_id} not found for merge_version increment")

    def increment_refine_version(self, insight_id: int):
        """
        Increment refine_version for an insight after successful refinement.
        
        Args:
            insight_id: The insight ID to update
        """
        for ins in self._library:
            if ins.insight_id == insight_id:
                ins.refine_version += 1
                return
        print(f"Warning: Insight {insight_id} not found for refine_version increment")

    def retrieve_insights_by_id(
        self,
        insight_ids: int | list[int],
        *,
        filter_fn: Optional[Callable[["Insight"], bool]] = None,
    ) -> list[dict]:
        """
        Retrieve insights by id(s) and return a list of dicts.
        If filter_fn is provided, only insights passing filter_fn(ins) are kept.
        """
        raw_ids = [insight_ids] if isinstance(insight_ids, int) else list(insight_ids)

        # Flatten nested ID lists
        ids_flat = []
        for iid in raw_ids:
            if isinstance(iid, list):
                ids_flat.extend(iid)
            else:
                ids_flat.append(iid)

        # For quick search, create a dict: id -> Insight
        id2ins = {ins.insight_id: ins for ins in self._library}

        # Gather matching insights in order, and convert to dict
        insights = []
        for iid in ids_flat:
            ins = id2ins.get(iid)
            if ins is None:
                continue

            #  filter_fn : A predicate to filter insights. Example: filter_fn=lambda ins: getattr(ins, "confidence", 0) > 0.7
            if filter_fn is not None and not filter_fn(ins):
                continue

            insights.append({
                "insight_id": ins.insight_id,
                "taxonomy": ins.taxonomy,
                "condition": ins.condition, 
                "explanation": ins.explanation,
                "example": ins.example,
            })

        return insights

    def retrieve_by_taxonomy(
        self,
        query_taxonomy: Dict[str, Dict[str, List[str]]],
        filter_fn: Optional[Callable[[Any], bool]] = None,
        include_task_id: bool = False
    ) -> List[Any]:
        """
        Retrieve all insights whose taxonomy contains at least one (stage, level-1, level-2) triple
        that appears in the provided query_taxonomy.
        query_taxonomy : dict
            Shape like:
            {
                "Domain Modeling": {
                    "Resource Allocation": ["Multi-Commodity Flow", ...]
                },
                "General Formulation": {
                    "Variable Definition": ["Continuous vs. Discrete Confusion", ...],
                    "Constraint Formulation": ["Incorrect Relational Operators"]
                },
                "Code Implementation": {
                    "Solver & API Syntax": ["Library Import/Reference Errors"]
                }
                }
        filter_fn : Callable[[Any], bool], optional
        A function that takes an insight and returns True if it should be included.
        If None, no filtering is applied.
        """

        # Build the target set of (stage, level-1, level-2) from query
        wanted = set()
        if isinstance(query_taxonomy, dict):
            for stage, lvl1_map in query_taxonomy.items():
                if not isinstance(lvl1_map, dict):
                    continue
                for lvl1, labels in lvl1_map.items():
                    if isinstance(labels, list):
                        # Query format: {stage: {level1: [level2_list]}}
                        for lbl in labels:
                            wanted.add((stage, lvl1, str(lbl)))
                    elif isinstance(labels, dict):
                        # Query format: {stage: {level1: {level2: value}}}
                        for lbl in labels.keys():
                            wanted.add((stage, lvl1, str(lbl)))
                    # if labels is neither list nor dict, ignore silently

        if not wanted:
            return []  # No valid query triples -> no results

        insights = []

        # Scan the library and test membership
        for ins in self._library:
            # Apply filter if provided
            if filter_fn is not None and not filter_fn(ins):
                continue
            tax = ins.taxonomy or {}
            matched = False

            for stage, lvl1_map in tax.items():
                if not isinstance(lvl1_map, dict):
                    continue
                for lvl1, lvl2_val in lvl1_map.items():
                    # Handle both list format (from query) and dict format (from library)
                    if isinstance(lvl2_val, list):
                        # Query format: {stage: {level1: [level2_list]}}
                        lvl2_list = lvl2_val
                    elif isinstance(lvl2_val, dict):
                        # Library format: {stage: {level1: {level2: {definition, condition}}}}
                        # Handle both dict with values and dict with None values
                        lvl2_list = [k for k, v in lvl2_val.items() if v is not None or k is not None]
                    else:
                        continue

                    # Check if any (stage, lvl1, lvl2) hits
                    for lbl in lvl2_list:
                        if (stage, lvl1, str(lbl)) in wanted:
                            matched = True
                            break
                if matched:
                    break

            if matched:
                # print("match taxonomy!")
                if include_task_id:
                    insights.append({
                        "insight_id": ins.insight_id,
                        "taxonomy": ins.taxonomy,
                        "condition": ins.condition,
                        "task_id": ins.task_id,
                        "iteration": ins.iteration
                    })
                else:
                    # Gather matching insights in order, and convert to dict
                    insights.append({
                        "insight_id": ins.insight_id,
                        "taxonomy": ins.taxonomy,
                        "condition": ins.condition
                    })

        return insights


    def add_insights(self, new_insights: list, iteration:int=None) -> None:
        # Current maximum id in the library (0 if empty)
        max_id = max(
            (ins.insight_id for ins in self._library if ins.insight_id is not None),
            default=0
        )

        # Optional: build a helper mapping from canonical taxonomy file
        def _load_canonical_level2_to_level1() -> Dict[str, tuple[str, str]] | dict:
            """
            Load the latest refined taxonomy and build a mapping:
                level-2 label -> (stage, level-1 label)

            This is used as a fallback when LLM output is missing the level-1 key
            (e.g., {"General Formulation": {"Redundant Auxiliary Variables": null}}
             while the canonical taxonomy is
             {"General Formulation": {"Objective Specification": {"Redundant Auxiliary Variables": null}}}).
            """
            canonical_path = "./data/experience_library/iterations/train_data_4o_flash/latest_taxonomy_refine_iter1.json"
            if not os.path.isfile(canonical_path):
                return {}

            try:
                with open(canonical_path, "r", encoding="utf-8") as f:
                    canonical_taxo = json.load(f)
            except Exception:
                return {}

            mapping: Dict[str, tuple[str, str]] = {}
            # canonical structure: {stage: {level1: {level2: ...}}}
            for stage, lvl1_map in canonical_taxo.items():
                if not isinstance(lvl1_map, dict):
                    continue
                for lvl1, lvl2_map in lvl1_map.items():
                    if not isinstance(lvl2_map, dict):
                        continue
                    for lvl2 in lvl2_map.keys():
                        # If duplicated, keep the first seen mapping
                        mapping.setdefault(str(lvl2), (stage, lvl1))
            return mapping

        canonical_level2_to_level1 = _load_canonical_level2_to_level1()

        for ins in new_insights:
            # Strictly validate taxonomy for newly produced insights; skip invalid ones.
            raw_taxo = ins.get("taxonomy", {})
            parsed_taxo = _parse_taxonomy_strict(raw_taxo)
            if parsed_taxo is None:
                print(
                    f"[WARNING] Skipping insight due to invalid taxonomy format "
                    f"(expected dict/JSON-dict; got {type(raw_taxo).__name__}): {str(raw_taxo)[:200]}"
                )
                continue
            # Skip empty / stage-missing taxonomy early to avoid retrieval verification blind spots.
            if not _taxonomy_has_any_label(parsed_taxo):
                print(
                    f"[WARNING] Skipping insight due to empty/invalid taxonomy content "
                    f"(no stage/label triples found). taxonomy_preview={str(raw_taxo)[:200]}"
                )
                continue

            max_id += 1                      # Assign the next incremental id

            # Use parsed taxonomy (dict) for downstream processing.
            taxo = parsed_taxo
            original_taxo = deepcopy(taxo)  # Save original for debugging
            
            # Check if taxonomy is already normalized (array format)
            # If so, convert back to dict format for repair and update_taxonomy
            is_normalized = False
            if isinstance(taxo, dict):
                for stage, lvl1_val in taxo.items():
                    if isinstance(lvl1_val, dict):
                        for lvl1_name, lvl2_val in lvl1_val.items():
                            if isinstance(lvl2_val, list):
                                is_normalized = True
                                break
                        if is_normalized:
                            break
                    if is_normalized:
                        break
            
            # If already normalized, convert back to dict format for processing
            if is_normalized:
                taxo = {}
                for stage, lvl1_val in original_taxo.items():
                    if isinstance(lvl1_val, dict):
                        taxo[stage] = {}
                        for lvl1_name, lvl2_list in lvl1_val.items():
                            if isinstance(lvl2_list, list):
                                # Convert array back to dict: {"Level2": {}}
                                taxo[stage][lvl1_name] = {l2_name: {} for l2_name in lvl2_list}
                            else:
                                taxo[stage][lvl1_name] = lvl2_list if isinstance(lvl2_list, dict) else {}


            if isinstance(taxo, dict) and canonical_level2_to_level1:
                repaired_taxo = deepcopy(taxo)
                for stage, lvl1_val in list(taxo.items()):
                    if not isinstance(lvl1_val, dict):
                        continue
                    for maybe_lvl1, maybe_lvl2_val in list(lvl1_val.items()):
                        # Case 1: value is not a dict (likely missing level-1; key is actually level-2)
                        if not isinstance(maybe_lvl2_val, dict):
                            lvl2_label = str(maybe_lvl1)
                            if lvl2_label in canonical_level2_to_level1:
                                canon_stage, canon_lvl1 = canonical_level2_to_level1[lvl2_label]
                                # Initialize nested dicts
                                repaired_taxo.setdefault(canon_stage, {})
                                if not isinstance(repaired_taxo[canon_stage], dict):
                                    repaired_taxo[canon_stage] = {}
                                repaired_taxo[canon_stage].setdefault(canon_lvl1, {})
                                if not isinstance(repaired_taxo[canon_stage][canon_lvl1], dict):
                                    repaired_taxo[canon_stage][canon_lvl1] = {}
                                # Move the entry as level-2 under canonical (stage, level-1)
                                repaired_taxo[canon_stage][canon_lvl1][lvl2_label] = None
                                # Remove the old, malformed entry
                                try:
                                    del repaired_taxo[stage][maybe_lvl1]
                                    if not repaired_taxo[stage]:
                                        del repaired_taxo[stage]
                                except Exception:
                                    pass
                        # Case 2: value is a dict but its inner values are lists like ["definition", "condition"]
                        # We still might want to rely on canonical mapping; handled later during normalization.
                taxo = repaired_taxo

            # Update library taxonomy BEFORE normalization (using dict format)
            # This ensures level-2 labels are properly added to library.taxonomy
            temp_ins_for_taxonomy = {"taxonomy": taxo}
            self.update_taxonomy(temp_ins_for_taxonomy)

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
        

            # Merge defaults with the incoming dict (incoming keys win if duplicated)
            # Preserve version fields if provided by caller (e.g., online/offline merge).
            enriched = {
                **ins,
                "insight_id":  max_id,
                "iteration": iteration,
                "merge_version": ins.get("merge_version", 0),
                "refine_version": ins.get("refine_version", 0),
                "occurrence":  [],  # list of retrieve counts per iteration
                "correctness": []   # list of success counts per iteration
            }

            self._library.append(Insight(enriched))


    @staticmethod
    def _update_one_stage(
        stage: Dict[str, Dict[str, str]],
        ins_label_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """
        Merge a single Level-1/Level-2 insight label into one stage of two-level taxonomy.
        
        Supports two input formats:
        1. Normalized (array format): {"Level1": ["Level2_1", "Level2_2", ...]}
        2. Original (dict format): {"Level1": {"Level2": null}} or {"Level1": {"Level2": {...}}}
        """

        def _find_ci_key(d: dict, key: str):
            """
            Return the existing key in dictionary that matches in a case-insensitive way
            """
            k_norm = str(key).casefold()
            for k in d.keys():
                if str(k).casefold() == k_norm:
                    return k
            return None
    
        for lvl1, lvl2_spec in (ins_label_dict or {}).items():
            # Find existing Level-1 key ignoring case
            l1_key = _find_ci_key(stage, lvl1)

            # If Level-1 doesn't exist, create it
            if l1_key is None:
                l1_key = lvl1
                stage[l1_key] = {}

            l2_dict = stage[l1_key]

            # Handle array format (normalized): {"Level1": ["Level2_1", "Level2_2", ...]}
            # This is the format after normalization in add_insights/generate_insights
            if isinstance(lvl2_spec, list):
                for l2_name in lvl2_spec:
                    if not isinstance(l2_name, str):
                        continue
                    # Find existing Level-2 key ignoring case
                    l2_key = _find_ci_key(l2_dict, l2_name)
                    # If not found, add it with empty dictionary
                    if l2_key is None:
                        l2_dict[l2_name] = {}
                    # If found, keep the original definition (no overwrite)
            
            # Handle dict format (original): {"Level1": {"Level2": null}} or {"Level1": {"Level2": {...}}}
            # This is the original format from LLM output
            elif isinstance(lvl2_spec, dict):
                for l2_name, info in lvl2_spec.items():
                    # Find existing Level-2 key ignoring case
                    l2_key = _find_ci_key(l2_dict, l2_name)
                    # If not found, add it with given information (definition + condition, empty dictionary if None)
                    if l2_key is None:
                        l2_dict[l2_name] = {} if info is None else info
                    else:
                        # If found, keep the original definition (no overwrite)
                        continue


    def update_taxonomy(self, new_labeled_insights: List[Dict[str, Any]]) -> None:
        """
        Update taxonomy dictionary based on a list of labeled insights
        """
        # Normalize single dict input to list
        if isinstance(new_labeled_insights, dict):
            new_labeled_insights = [new_labeled_insights]

        for ins in new_labeled_insights or []:
            ins_taxo = ins.get("taxonomy", {})
            if not isinstance(ins_taxo, dict):
                continue

            # Domain Modeling
            if "Domain Modeling" in ins_taxo and isinstance(ins_taxo["Domain Modeling"], dict):
                self._update_one_stage(self.taxonomy.setdefault("Domain Modeling", {}),
                                    ins_taxo["Domain Modeling"])

            # General Formulation
            if "General Formulation" in ins_taxo and isinstance(ins_taxo["General Formulation"], dict):
                self._update_one_stage(self.taxonomy.setdefault("General Formulation", {}),
                                    ins_taxo["General Formulation"])

            # Code Implementation
            if "Code Implementation" in ins_taxo and isinstance(ins_taxo["Code Implementation"], dict):
                self._update_one_stage(self.taxonomy.setdefault("Code Implementation", {}),
                                    ins_taxo["Code Implementation"])
    

    def replace_merged_insights(self, existing_insights: List[dict]) -> None:
        """
        Replace existing insights that were merged with new merged insights.
        This removes the old insights from the library.
        
        Args:
            existing_insights: List of existing insights that were merged
        """
        # Get insight IDs that were merged (from existing_insights)
        merged_insight_ids = [ins["insight_id"] for ins in existing_insights]
        
        # Remove the merged insights from library
        self._library = [ins for ins in self._library if ins.insight_id not in merged_insight_ids]
        
        print(f"Removed {len(merged_insight_ids)} existing insights that were merged: {merged_insight_ids}")

    def remove_insights_by_ids(self, insight_ids: list[int]) -> None:
        """
        Remove insights from the library by their IDs.
        """
        if not insight_ids:
            return
        
        insight_ids_set = set(insight_ids)
        removed_count = 0
        original_count = len(self._library)
        
        self._library = [ins for ins in self._library if ins.insight_id not in insight_ids_set]
        
        removed_count = original_count - len(self._library)
        if removed_count > 0:
            print(f"Removed {removed_count} insight(s) with IDs: {insight_ids}")

    def add_insights_update_ids(self, new_insights: list, iter: int, lock=None):
        """
        Add new insights to library and update their insight_ids to match the IDs assigned by add_insights.
        This method handles thread-safe operations when a lock is provided.
        """
        def _add_and_update():
            # Filter out insights with invalid/empty taxonomy BEFORE adding; keep list in sync for downstream verification.
            valid = []
            for ins in list(new_insights or []):
                raw_taxo = ins.get("taxonomy", {})
                parsed_taxo = _parse_taxonomy_strict(raw_taxo)
                if parsed_taxo is None or not _taxonomy_has_any_label(parsed_taxo):
                    print(
                        f"[WARNING] add_insights_update_ids: skipping insight due to invalid/empty taxonomy "
                        f"(taxonomy_type={type(raw_taxo).__name__})."
                    )
                    continue
                # Normalize to parsed dict so add_insights won't re-parse JSON strings inconsistently.
                ins["taxonomy"] = parsed_taxo
                valid.append(ins)

            # Mutate in-place so callers don't verify/remove insights that were never added.
            if new_insights is not None:
                new_insights[:] = valid

            if not new_insights:
                return

            max_id_before = max(
                (ins.insight_id for ins in self._library if ins.insight_id is not None),
                default=0
            )
            for ins in new_insights:
                self.add_insights([ins], iter)
            # Update insight_id in new_insights to match the IDs assigned by add_insights
            newly_added = sorted(
                [ins for ins in self._library 
                 if ins.insight_id is not None and ins.insight_id > max_id_before],
                key=lambda x: x.insight_id
            )
            for idx, ins in enumerate(new_insights):
                if idx < len(newly_added):
                    ins['insight_id'] = newly_added[idx].insight_id
        
        if lock is not None:
            with lock:
                _add_and_update()
        else:
            _add_and_update()

    def remove_unverified_insights(self, new_insights: list, is_verify: bool, verified_insights: list, lock=None):
        """
        Remove unverified insights from the library based on verification results.
        """
        def _remove():
            new_insights_ids = {ins.get('insight_id') for ins in new_insights}
            if is_verify:
                # All insights verified: keep all (already in library)
                pass
            elif verified_insights:
                # Partial retrieval: keep only verified insights, remove unverified ones
                verified_ids = {ins.get('insight_id') for ins in verified_insights}
                unverified_ids = [ins_id for ins_id in new_insights_ids if ins_id not in verified_ids]
                if unverified_ids:
                    self.remove_insights_by_ids(unverified_ids)
            else:
                # No insights verified: remove all new_insights
                if new_insights_ids:
                    self.remove_insights_by_ids(list(new_insights_ids))
        
        if lock is not None:
            with lock:
                _remove()
        else:
            _remove()

    def merge_into_library(
        self,
        all_merged_ids: List[List[int]],
        all_merged_insights: List[dict],
        lib_base: "ExperienceLibrary"
    ) -> "ExperienceLibrary":
        """
        Produce a new library variant by replacing old insights with new merged insights
        """

        # Cache an id → Insight map before deletions
        cache_id2ins = {ins.insight_id: ins for ins in lib_base._library}

        # Remove every insight that will be merged
        ids_to_remove = set(itertools.chain.from_iterable(all_merged_ids))
        lib_base._library = [
            ins for ins in lib_base._library
            if ins.insight_id not in ids_to_remove
        ]

        current_max_id = max((ins.insight_id for ins in lib_base._library), default=0)

        # Append each newly merged insight
        for ids_group, merged_ins in zip(all_merged_ids, all_merged_insights):
            current_max_id += 1

            task_ids_set = set()

            for parent_id in ids_group:
                parent_ins = cache_id2ins.get(parent_id)
                # source task ids
                if parent_ins.task_id is not None:
                    if isinstance(parent_ins.task_id, list):
                        task_ids_set.update(parent_ins.task_id)
                    else:
                        task_ids_set.add(parent_ins.task_id)

            source_task_ids  = sorted(task_ids_set) if task_ids_set else None

            # Get max merge_version from parent insights
            parent_merge_versions = [parent_ins.merge_version for parent_ins in [cache_id2ins.get(pid) for pid in ids_group] if parent_ins]
            new_merge_version = max(parent_merge_versions, default=0) + 1 if parent_merge_versions else 1

            # Inherit occurrence and correctness from parents (merge lists)
            parent_occurrence = []
            parent_correctness = []
            for parent_id in ids_group:
                parent_ins = cache_id2ins.get(parent_id)
                if parent_ins:
                    # Extend with parent's occurrence and correctness lists
                    if isinstance(parent_ins.occurrence, list):
                        parent_occurrence.extend(parent_ins.occurrence)
                    if isinstance(parent_ins.correctness, list):
                        parent_correctness.extend(parent_ins.correctness)

            merged_insight_enriched = {
                "insight_id":  current_max_id,
                "taxonomy": merged_ins.get("taxonomy", ""),
                "condition":   merged_ins.get("condition", ""),
                "explanation": merged_ins.get("explanation", ""),
                "example":     merged_ins.get("example", ""),
                "distribution": {"positive": [], "negative": [], "unretrieved": [], "irrelevant": [], "invalid": []},
                # default / inherited fields
                "task_id":     source_task_ids,
                "merge_version": new_merge_version,
                "refine_version": 0,  # Reset refine_version for merged insight
                "occurrence":  parent_occurrence,  # Inherit from parents
                "correctness": parent_correctness,  # Inherit from parents
            }

            lib_base._library.append(Insight(merged_insight_enriched))

        return lib_base

# Usage example
if __name__ ==  "__main__":
    # Load the library
    lib_path = "./data/experience_library/iterations/train_data_4o/library_diag_iter1.json"
    taxo_path = "./data/experience_library/iterations/train_data_4o/latest_taxonomy_diag_iter1.json"
    library = ExperienceLibrary.from_json_file(
        library_path = lib_path,
        taxonomy_path = taxo_path)

    print(f"Library loaded with {len(library)} insights")
    
    # Test cases
    test_cases = [
        {
            'General Formulation': {
                'Variable Definition': {
                    'Continuous vs. Discrete Confusion': {
                        'definition': 'Choose integer/binary for indivisible items; continuous for divisible flows.',
                        'condition': 'Applies when decision quantities in the problem represent indivisible counts or choices versus divisible amounts such as flows.'
                    }
                }
            }
        },
        {
            'General Formulation': {
                'Explicit Bounds': None
            }
        },
        {
            'General Formulation': {
                'Unit Inconsistency': {
                    'definition': 'Keep all terms in compatible units to avoid 1000x errors.',
                    'condition': 'Applies when input data come from different unit systems or incompatible measurement scales.'
                }
            }
        },
        {
            'General Formulation': {
                'Variable Definition': {
                    'Continuous vs. Discrete Confusion': None
                }
            }
        },
        {
            'General Formulation': {
                'Constraint Formulation': {
                    'Incorrect Relational Operators': None
                }
            }
        }
    ]
    
    print("\n" + "="*60)
    print("Testing retrieve_by_taxonomy with different formats")
    print("="*60)
    
    for i, test_taxonomy in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Query taxonomy: {test_taxonomy}")
        
        # Test retrieve_by_taxonomy
        results = library.retrieve_by_taxonomy(
            query_taxonomy=test_taxonomy, 
            include_task_id=True
        )
        
        print(f"Found {len(results)} matching insights")
        
        if results:
            print("Matching insights:")
            for j, result in enumerate(results[:3]):  # Show first 3 results
                print(f"  {j+1}. Insight ID: {result.get('insight_id', 'N/A')}")
                print(f"     Task ID: {result.get('task_id', 'N/A')}")
                print(f"     Taxonomy: {result.get('taxonomy', {})}")
        else:
            print("  No matching insights found")
        
        print("-" * 40)
    
    print("\nTest completed!")
