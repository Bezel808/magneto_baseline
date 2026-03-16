#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from magneto import Magneto


DATASET_ROOT_DEFAULT = Path("/home/mengshi/table_quality/datasets_joint_discovery_integration")
DATASETS_DEFAULT = ["wikidbs_1218", "santos_benchmark_1218", "magellan_1218"]


@dataclass
class SplitResult:
    y_true: np.ndarray
    y_score: np.ndarray
    elapsed_sec: float
    table_pairs: int
    rows: int


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_split_labels(dataset_root: Path, split: str) -> pd.DataFrame:
    path = dataset_root / "label_plus" / "schema_matching" / f"{split}.csv"
    df = pd.read_csv(
        path,
        usecols=[
            "table_name_1",
            "renamed_column_name_1",
            "table_name_2",
            "renamed_column_name_2",
            "label",
        ],
    )
    return df


def _evaluate_split(
    dataset_root: Path,
    df_split: pd.DataFrame,
    matcher_kwargs: dict[str, object],
    cache: OrderedDict[str, pd.DataFrame],
    max_cache_tables: int = 32,
    reset_every_pairs: int = 40,
) -> SplitResult:
    datalake = dataset_root / "datalake_plus"
    pairs = df_split[["table_name_1", "table_name_2"]].drop_duplicates()
    pair_to_rows: dict[tuple[str, str], pd.DataFrame] = {
        (a, b): g for (a, b), g in df_split.groupby(["table_name_1", "table_name_2"], sort=False)
    }

    y_true: list[int] = []
    y_score: list[float] = []
    matcher: Magneto | None = None

    t0 = time.perf_counter()
    for idx, row in enumerate(pairs.itertuples(index=False), start=1):
        if matcher is None or (idx - 1) % reset_every_pairs == 0:
            matcher = Magneto(**matcher_kwargs)

        t1 = row.table_name_1
        t2 = row.table_name_2

        if t1 not in cache:
            cache[t1] = pd.read_csv(datalake / t1)
            while len(cache) > max_cache_tables:
                cache.popitem(last=False)
        else:
            cache.move_to_end(t1)
        if t2 not in cache:
            cache[t2] = pd.read_csv(datalake / t2)
            while len(cache) > max_cache_tables:
                cache.popitem(last=False)
        else:
            cache.move_to_end(t2)

        matches = matcher.get_matches(cache[t1], cache[t2])
        score_map = {k: float(v) for k, v in matches.items()}
        group = pair_to_rows[(t1, t2)]
        for r in group.itertuples(index=False):
            key = (("source", r.renamed_column_name_1), ("target", r.renamed_column_name_2))
            y_true.append(int(r.label))
            y_score.append(score_map.get(key, 0.0))

        if idx % 50 == 0 or idx == len(pairs):
            print(f"[{_now()}] table_pairs {idx}/{len(pairs)}", flush=True)

    elapsed = time.perf_counter() - t0
    return SplitResult(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_score=np.asarray(y_score, dtype=np.float64),
        elapsed_sec=elapsed,
        table_pairs=len(pairs),
        rows=len(df_split),
    )


def _find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, n_steps: int = 1001) -> float:
    best_t = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.0, 1.0, n_steps)
    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auc"] = 0.0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Magneto baseline for schema matching on *_1218 datasets")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--datasets", nargs="+", default=DATASETS_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=Path("runs") / "magneto_sm_1218")
    parser.add_argument("--mode", default="header_values_default")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--embedding-threshold", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / ts
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{_now()}] run_root={run_root}", flush=True)
    summary: dict[str, dict] = {}

    matcher_kwargs: dict[str, object] = {
        "encoding_mode": args.mode,
        "topk": args.topk,
        "embedding_threshold": args.embedding_threshold,
        "use_gpt_reranker": False,
    }
    if args.cpu:
        matcher_kwargs["device"] = "cpu"

    for ds in args.datasets:
        ds_root = args.dataset_root / ds
        ds_log = logs_dir / f"{ds}.log"
        ds_t0 = time.perf_counter()
        print(f"[{_now()}] [DATASET={ds}] start", flush=True)

        with ds_log.open("w", encoding="utf-8") as lf:
            lf.write(f"[{_now()}] dataset_root={ds_root}\n")

        validate_df = _load_split_labels(ds_root, "validate")
        test_df = _load_split_labels(ds_root, "test")

        cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        val_res = _evaluate_split(ds_root, validate_df, matcher_kwargs, cache)
        best_t = _find_best_threshold(val_res.y_true, val_res.y_score)
        val_metrics = _metrics(val_res.y_true, val_res.y_score, best_t)

        test_res = _evaluate_split(ds_root, test_df, matcher_kwargs, cache)
        test_metrics = _metrics(test_res.y_true, test_res.y_score, best_t)

        total = time.perf_counter() - ds_t0
        summary[ds] = {
            "dataset": ds,
            "dataset_root": str(ds_root),
            "validate": {
                "rows": val_res.rows,
                "table_pairs": val_res.table_pairs,
                "elapsed_sec": round(val_res.elapsed_sec, 3),
                "metrics": val_metrics,
            },
            "test": {
                "rows": test_res.rows,
                "table_pairs": test_res.table_pairs,
                "elapsed_sec": round(test_res.elapsed_sec, 3),
                "metrics": test_metrics,
            },
            "timing_seconds": {"total": round(total, 3)},
            "log_path": str(ds_log),
        }

        with ds_log.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(summary[ds], ensure_ascii=False, indent=2))
            lf.write("\n")

        # Persist incremental summary so partial results remain available if a later dataset crashes.
        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{_now()}] [DATASET={ds}] done: test_f1={test_metrics['f1']:.6f}", flush=True)

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[{_now()}] [DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
