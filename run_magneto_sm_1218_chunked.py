#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from magneto import Magneto


DATASET_ROOT_DEFAULT = Path("/home/mengshi/table_quality/datasets_joint_discovery_integration")
DATASETS_DEFAULT = ["wikidbs_1218", "santos_benchmark_1218", "magellan_1218"]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_split_labels(dataset_root: Path, split: str) -> pd.DataFrame:
    path = dataset_root / "label_plus" / "schema_matching" / f"{split}.csv"
    return pd.read_csv(
        path,
        usecols=[
            "table_name_1",
            "renamed_column_name_1",
            "table_name_2",
            "renamed_column_name_2",
            "label",
        ],
    )


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


def _find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, n_steps: int = 1001) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.0, 1.0, n_steps):
        y_pred = (y_score >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _worker_mode(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    df_split = _load_split_labels(dataset_root, args.split)
    pairs = df_split[["table_name_1", "table_name_2"]].drop_duplicates().reset_index(drop=True)
    chunk_pairs = pairs.iloc[args.pair_start : args.pair_end]
    grouped = {
        (a, b): g
        for (a, b), g in df_split.groupby(["table_name_1", "table_name_2"], sort=False)
    }

    matcher = Magneto(
        encoding_mode=args.mode,
        topk=args.topk,
        embedding_threshold=args.embedding_threshold,
        use_gpt_reranker=False,
        device=args.device,
    )

    datalake = dataset_root / "datalake_plus"
    cache: dict[str, pd.DataFrame] = {}
    y_true: list[int] = []
    y_score: list[float] = []

    t0 = time.perf_counter()
    for row in chunk_pairs.itertuples(index=False):
        t1 = row.table_name_1
        t2 = row.table_name_2
        if t1 not in cache:
            cache[t1] = pd.read_csv(datalake / t1)
        if t2 not in cache:
            cache[t2] = pd.read_csv(datalake / t2)

        matches = matcher.get_matches(cache[t1], cache[t2])
        score_map = {k: float(v) for k, v in matches.items()}
        g = grouped[(t1, t2)]
        for r in g.itertuples(index=False):
            key = (("source", r.renamed_column_name_1), ("target", r.renamed_column_name_2))
            y_true.append(int(r.label))
            y_score.append(score_map.get(key, 0.0))

    elapsed = time.perf_counter() - t0
    out = {
        "pair_start": args.pair_start,
        "pair_end": args.pair_end,
        "pairs": int(len(chunk_pairs)),
        "rows": int(len(y_true)),
        "elapsed_sec": float(elapsed),
        "y_true": y_true,
        "y_score": y_score,
    }
    Path(args.worker_out).write_text(json.dumps(out), encoding="utf-8")
    return 0


def _run_split_chunked(
    dataset_root: Path,
    split: str,
    mode: str,
    topk: int,
    embedding_threshold: float,
    device: str,
    max_pairs_per_worker: int,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    df_split = _load_split_labels(dataset_root, split)
    pairs = df_split[["table_name_1", "table_name_2"]].drop_duplicates().reset_index(drop=True)
    n_pairs = len(pairs)

    y_true_all: list[int] = []
    y_score_all: list[float] = []
    total_elapsed = 0.0
    grouped = {
        (a, b): g
        for (a, b), g in df_split.groupby(["table_name_1", "table_name_2"], sort=False)
    }

    this_py = Path(__file__).resolve()
    failed_pairs = 0
    failed_chunks: list[tuple[int, int]] = []
    chunks: list[tuple[int, int]] = []
    for start in range(0, n_pairs, max_pairs_per_worker):
        chunks.append((start, min(start + max_pairs_per_worker, n_pairs)))

    def _run_worker(start: int, end: int) -> tuple[int, dict | None]:
        with tempfile.NamedTemporaryFile(prefix=f"magneto_{split}_{start}_{end}_", suffix=".json", delete=False) as tf:
            worker_out = tf.name

        cmd = [
            sys.executable,
            str(this_py),
            "--worker",
            "--dataset-root",
            str(dataset_root),
            "--split",
            split,
            "--pair-start",
            str(start),
            "--pair-end",
            str(end),
            "--mode",
            mode,
            "--topk",
            str(topk),
            "--embedding-threshold",
            str(embedding_threshold),
            "--device",
            device,
            "--worker-out",
            worker_out,
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
        env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
        env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")

        print(f"[{_now()}] split={split} chunk={start}:{end} start", flush=True)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            Path(worker_out).unlink(missing_ok=True)
            return r.returncode, None
        blob = json.loads(Path(worker_out).read_text(encoding="utf-8"))
        Path(worker_out).unlink(missing_ok=True)
        return 0, blob

    # Phase 1: fixed-size chunks.
    for start, end in chunks:
        rc, blob = _run_worker(start, end)
        if rc != 0:
            failed_chunks.append((start, end))
            print(
                f"[{_now()}] split={split} chunk={start}:{end} failed exit={rc}; defer to per-pair fallback",
                flush=True,
            )
            continue
        assert blob is not None
        y_true_all.extend(blob["y_true"])
        y_score_all.extend(blob["y_score"])
        total_elapsed += float(blob["elapsed_sec"])
        print(f"[{_now()}] split={split} chunk={start}:{end} done", flush=True)

    # Phase 2: fallback only on failed chunks, one pair per worker.
    for start, end in failed_chunks:
        for i in range(start, end):
            rc, blob = _run_worker(i, i + 1)
            if rc != 0:
                pair_row = pairs.iloc[i]
                key = (pair_row["table_name_1"], pair_row["table_name_2"])
                g = grouped[key]
                y_true_all.extend([int(x) for x in g["label"].tolist()])
                y_score_all.extend([0.0] * len(g))
                failed_pairs += 1
                print(
                    f"[{_now()}] split={split} pair={i}:{i+1} hard-failed exit={rc}; fallback score=0 rows={len(g)}",
                    flush=True,
                )
                continue
            assert blob is not None
            y_true_all.extend(blob["y_true"])
            y_score_all.extend(blob["y_score"])
            total_elapsed += float(blob["elapsed_sec"])
            print(f"[{_now()}] split={split} pair={i}:{i+1} done", flush=True)

    if failed_pairs > 0:
        print(f"[{_now()}] split={split} failed_pairs={failed_pairs} (fallback score=0)", flush=True)

    return (
        np.asarray(y_true_all, dtype=np.int64),
        np.asarray(y_score_all, dtype=np.float64),
        int(len(df_split)),
        int(n_pairs),
        float(total_elapsed),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Magneto SM on *_1218 with chunked workers")
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    p.add_argument("--datasets", nargs="+", default=DATASETS_DEFAULT)
    p.add_argument("--output-root", type=Path, default=Path("runs") / "magneto_sm_1218")
    p.add_argument("--mode", default="header_values_default")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--embedding-threshold", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-pairs-per-worker", type=int, default=30)

    p.add_argument("--worker", action="store_true")
    p.add_argument("--split", default="")
    p.add_argument("--pair-start", type=int, default=0)
    p.add_argument("--pair-end", type=int, default=0)
    p.add_argument("--worker-out", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        raise SystemExit(_worker_mode(args))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / ts
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.json"
    summary: dict[str, dict] = {}

    print(f"[{_now()}] run_root={run_root}", flush=True)
    for ds in args.datasets:
        ds_root = args.dataset_root / ds
        print(f"[{_now()}] [DATASET={ds}] start", flush=True)
        t0 = time.perf_counter()

        val_y, val_s, val_rows, val_pairs, val_elapsed = _run_split_chunked(
            ds_root, "validate", args.mode, args.topk, args.embedding_threshold, args.device, args.max_pairs_per_worker
        )
        best_t = _find_best_threshold(val_y, val_s)
        val_metrics = _metrics(val_y, val_s, best_t)

        test_y, test_s, test_rows, test_pairs, test_elapsed = _run_split_chunked(
            ds_root, "test", args.mode, args.topk, args.embedding_threshold, args.device, args.max_pairs_per_worker
        )
        test_metrics = _metrics(test_y, test_s, best_t)

        summary[ds] = {
            "dataset": ds,
            "dataset_root": str(ds_root),
            "validate": {
                "rows": val_rows,
                "table_pairs": val_pairs,
                "elapsed_sec": round(val_elapsed, 3),
                "metrics": val_metrics,
            },
            "test": {
                "rows": test_rows,
                "table_pairs": test_pairs,
                "elapsed_sec": round(test_elapsed, 3),
                "metrics": test_metrics,
            },
            "timing_seconds": {"total": round(time.perf_counter() - t0, 3)},
            "log_path": str(logs_dir / f"{ds}.log"),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{_now()}] [DATASET={ds}] done: test_f1={test_metrics['f1']:.6f}", flush=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[{_now()}] [DONE] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
