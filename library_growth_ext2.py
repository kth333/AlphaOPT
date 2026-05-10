"""
Failure-Driven Library Growth (FDLG) — Extension 2.

Mines failed tasks from a calibration partition of the eval set and feeds them
to AlphaOPT's *existing* `library_diagnosis` + `library_refinement` modules to
grow the trained library beyond iter=2. The inference pipeline is unchanged at
deployment — only the library content changes.

Usage:
    python library_growth_ext2.py [config_path]

The default config is `growth_config_ext2.yaml`.
"""
from __future__ import annotations

import os
import sys
import json
import time
import copy
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

# Force UTF-8 for stdout/stderr on Windows so library prints don't crash on
# non-ASCII characters (✓, ✅, etc.) emitted by library_diagnosis.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from omegaconf import OmegaConf

from src.dataloader import DataLoader, Task
from src.utils import cal_time_cost
from src.train_eval_utils import save_checkpoint, print_training_metrics_summary
from src.experience_library import ExperienceLibrary
from src.llm_programmer import ProgramGenerator
from src.llm_diagnostic import ProgramDiagnostic
from src.llm_extractor import InsightExtractor
from src.llm_retriever import LibraryRetrieval
from src.llm_evolver import LibraryEvolution

from library_diagnosis import run_library_diagnosis
from library_refinement import run_library_refinement


# ---------------------------------------------------------------------------
# Step 1: Failure mining
# ---------------------------------------------------------------------------

def mine_failure_records(
    calibration_datasets: List[str],
    calibration_runs: List[str],
    baseline_run_dir_template: str,
    failure_filter: str,
) -> List[Tuple[str, dict]]:
    """
    Walk the existing baseline `tasks_record_lib.json` files for the calibration
    datasets and return the failure records.

    Returns
    -------
    list of (dataset_name, raw_record_dict)
        Deduplicated across runs by (dataset, task_id).
    """
    seen_keys: set = set()
    out: List[Tuple[str, dict]] = []

    for ds in calibration_datasets:
        for run in calibration_runs:
            path = baseline_run_dir_template.replace("${dataset}", ds).replace("${run}", run)
            p = Path(path)
            if not p.exists():
                print(f"[mine] missing: {p} — skipping")
                continue

            with open(p, encoding="utf-8") as f:
                records = json.load(f)

            for rec in records:
                status_lst = rec.get("output_status") or [None]
                status = status_lst[0] if status_lst else None
                if status == "optimal":
                    continue

                # Failure-mode filter
                ri = rec.get("retrieved_insights") or []
                has_ret = any(isinstance(inner, list) and inner for inner in ri)

                if failure_filter == "formulation_with_retrieval":
                    keep = (status == "not_optimal") and has_ret
                elif failure_filter == "all_failures":
                    keep = True
                else:
                    raise ValueError(
                        f"Unknown failure_filter: {failure_filter!r} "
                        f"(expected 'formulation_with_retrieval' | 'all_failures')"
                    )

                if not keep:
                    continue

                key = (ds, rec.get("task_id"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append((ds, rec))

    print(f"[mine] {len(out)} unique failed task-records across "
          f"{len(calibration_datasets)} dataset(s) × {len(calibration_runs)} run(s)")
    return out


# ---------------------------------------------------------------------------
# Step 2: Build a DataLoader of fresh Task objects from the failed records
# ---------------------------------------------------------------------------

def _load_test_data(dataset: str, test_data_path_template: str) -> dict:
    """Load test data and index it by task_id for fast lookup."""
    path = test_data_path_template.replace("${dataset}", dataset)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["task_id"]: item for item in data}


def build_failure_dataloader(
    failure_records: List[Tuple[str, dict]],
    test_data_path_template: str,
) -> DataLoader:
    """
    Construct a DataLoader of Task objects for the failed calibration tasks.

    Tasks are loaded fresh from the canonical test JSONs so that fields like
    `description`, `ground_truth`, `tag`, and (when available) `correct_program`
    come from the source rather than the run-record summary. Progress fields
    are reset so library_diagnosis re-runs the pipeline cleanly.
    """
    test_indices: dict[str, dict] = {}
    tasks: list[Task] = []

    for dataset, rec in failure_records:
        if dataset not in test_indices:
            test_indices[dataset] = _load_test_data(dataset, test_data_path_template)

        tid = rec.get("task_id")
        src = test_indices[dataset].get(tid)
        if src is None:
            print(f"[build] WARN: task_id {tid!r} not in {dataset} test data — skipping")
            continue

        gt = src.get("ground_truth")
        if gt is None or (isinstance(gt, float) and gt != gt):
            print(f"[build] WARN: task_id {tid!r} missing ground_truth — skipping")
            continue
        if isinstance(gt, str):
            try:
                gt = float(gt)
            except ValueError:
                print(f"[build] WARN: task_id {tid!r} non-numeric ground_truth — skipping")
                continue

        # Namespace task_id by dataset to avoid collisions in output dirs and
        # in the library's distribution lists. e.g. "MAM_A042".
        # NOTE: separator must be filesystem-safe (no ':' on Windows).
        namespaced_id = f"{dataset[:3].upper()}_{tid}"

        t = Task(
            task_id=namespaced_id,
            desc=src.get("description"),
            ground_truth=gt,
            formulation=src.get("formulation"),
            correct_program=src.get("correct_program"),
            tag=src.get("tag"),
            cluster=src.get("cluster"),
        )
        # Progress fields start fresh so diagnosis sees this as a new attempt.
        # (Task.__init__ already initializes them; explicit for clarity.)
        t.success_count = 0
        t.confidence = 0
        t.output_status = []
        t.fail_to_execute = 0
        t.fail_to_verify = 0
        t.retri_ins_lst = []
        tasks.append(t)

    print(f"[build] DataLoader of {len(tasks)} tasks ready for diagnosis")
    return DataLoader(task_list=tasks)


# ---------------------------------------------------------------------------
# Step 3: Orchestrate diagnosis + refinement
# ---------------------------------------------------------------------------

def _get_template_str(cfg, key: str) -> str:
    """
    Extract a string field that contains ${dataset} / ${run} placeholders.

    OmegaConf eagerly tries to interpolate any ${...} on attribute access, even
    for keys we intend to substitute manually. Going through to_container with
    resolve=False yields the raw literal, which is what we need.
    """
    raw = OmegaConf.to_container(cfg, resolve=False)
    if key not in raw:
        raise KeyError(f"Config field {key!r} missing")
    return str(raw[key])


def run_growth(config_path: str = "growth_config_ext2.yaml"):
    # Do NOT call OmegaConf.resolve(cfg) here — the config intentionally
    # contains `${dataset}` / `${run}` template placeholders that we
    # substitute manually in mine_failure_records / build_failure_dataloader.
    cfg = OmegaConf.load(config_path)

    # ---- Output dirs -------------------------------------------------------
    Path(cfg.file_paths.lib_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.file_paths.train_output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(cfg.file_paths.metrics_log_path)
    if not metrics_path.exists():
        metrics_path.write_text("[]", encoding="utf-8")

    # ---- Step 1: mine failures --------------------------------------------
    print("=" * 70)
    print("[FDLG] Step 1/4: Mining calibration failures")
    print("=" * 70)
    failure_records = mine_failure_records(
        calibration_datasets=list(cfg.calibration_datasets),
        calibration_runs=list(cfg.calibration_runs),
        baseline_run_dir_template=_get_template_str(cfg, "baseline_run_dir_template"),
        failure_filter=cfg.failure_filter,
    )

    if not failure_records:
        print("[FDLG] No failed tasks found in calibration set. Aborting.")
        return

    # ---- Step 2: build DataLoader ------------------------------------------
    print("\n" + "=" * 70)
    print("[FDLG] Step 2/4: Building DataLoader from failed tasks")
    print("=" * 70)
    failure_loader = build_failure_dataloader(
        failure_records=failure_records,
        test_data_path_template=_get_template_str(cfg, "test_data_path_template"),
    )

    # Save the failed-task list as a record file so it's auditable
    failure_loader.save_as_json(cfg.file_paths.train_data_path)
    print(f"[FDLG] Saved failure DataLoader to {cfg.file_paths.train_data_path}")

    # ---- Step 3: load library + initialize agents --------------------------
    print("\n" + "=" * 70)
    print("[FDLG] Step 3/4: Loading library + initializing agents")
    print("=" * 70)
    library = ExperienceLibrary.from_json_file(
        library_path=cfg.source_library_path,
        taxonomy_path=cfg.source_taxonomy_path,
    )
    print(f"[FDLG] Source library: {len(library)} insights at "
          f"{cfg.source_library_path}")

    llm_opt = ProgramGenerator(
        model=cfg.base_model, service=cfg.base_service, temperature=0)
    llm_diag = ProgramDiagnostic(
        model=cfg.advanced_model, service=cfg.advanced_service, temperature=0)
    llm_ins = InsightExtractor(
        model=cfg.advanced_model, service=cfg.advanced_service, temperature=0.7)
    llm_retri = LibraryRetrieval(
        lib=library, model=cfg.base_model, service=cfg.base_service, temperature=0)

    # ---- Step 4: diagnosis + refinement ------------------------------------
    print("\n" + "=" * 70)
    print(f"[FDLG] Step 4/4: Running library_diagnosis (iter={cfg.start_iter})")
    print("=" * 70)
    growth_t0 = time.time()

    with open(cfg.file_paths.metrics_log_path, "r", encoding="utf-8") as f:
        metrics_log = json.load(f)

    # Diagnosis: mutates `library` in place, queues new insights through
    # online merge, populates distribution.{positive, negative, ...}.
    diag_metrics = run_library_diagnosis(
        iter=cfg.start_iter,
        train_tasks=failure_loader,
        llm_retri=llm_retri,
        llm_opt=llm_opt,
        llm_diag=llm_diag,
        llm_ins=llm_ins,
        library=library,
        params=cfg.params,
        paths=cfg.file_paths,
        max_workers=int(cfg.get("diagnosis_max_workers", 6)),
    )
    metrics_log.append(diag_metrics)
    save_checkpoint(
        library=library, tasks=failure_loader, metrics=metrics_log,
        paths=cfg.file_paths, suffix=f"diag_iter{cfg.start_iter}",
    )
    print(f"[FDLG] Post-diagnosis library size: {len(library)} insights")

    # Refinement
    print("\n" + "=" * 70)
    print(f"[FDLG] Running library_refinement (iter={cfg.start_iter})")
    print("=" * 70)
    llm_evolve = LibraryEvolution(
        lib=library, model=cfg.advanced_model,
        service=cfg.advanced_service, temperature=0.7)

    (
        refined_library,
        avg_refinement_rate,
        token_usage_delta,
        duration_min,
        refined_ins_num,
        refinement_success_rate,
        refinement_success_num,
    ) = run_library_refinement(
        iter=cfg.start_iter,
        tasks=failure_loader,
        config=cfg,
        llm_evolve=llm_evolve,
        verbose=False,
        save_data=True,
        output_path=cfg.file_paths.train_output_dir,
        max_workers=int(cfg.get("refinement_max_workers", 4)),
    )

    last = metrics_log[-1]
    last["refinement_avg_gain"] = round(float(avg_refinement_rate), 3)
    last["refined_ins_num"] = int(refined_ins_num)
    last["refinement_success_rate"] = round(float(refinement_success_rate), 3)
    last["refinement_success_num"] = int(refinement_success_num)
    last["refinement_proposed_num"] = int(refined_ins_num)
    last["refinement_token_usage"] = token_usage_delta
    last["library_refinement_duration (min)"] = round(float(duration_min), 3)

    # Save the grown library under both a versioned name (refine_iter3) and
    # a stable alias (grown) for the eval config.
    save_checkpoint(
        library=refined_library, tasks=None, metrics=metrics_log,
        paths=cfg.file_paths, suffix=f"refine_iter{cfg.start_iter}",
    )
    refined_library.save(f"{cfg.file_paths.lib_dir}/library_grown.json")
    refined_library.save_taxonomy(
        f"{cfg.file_paths.lib_dir}/latest_taxonomy_grown.json")

    print(f"\n[FDLG] Final library size: {len(refined_library)} insights")
    print(f"[FDLG] Saved grown library to {cfg.file_paths.lib_dir}/library_grown.json")

    cal_time_cost(growth_t0, "FDLG total")
    print_training_metrics_summary(metrics_log)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "growth_config_ext2.yaml"
    run_growth(cfg_path)
