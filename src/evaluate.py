from __future__ import annotations

"""Post-hoc evaluation script that downloads run data from WandB and
produces per-run as well as cross-run analyses/figures.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind

################################################################################
# I/O helpers                                                                  #
################################################################################

def _save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

################################################################################
# Plotting utilities                                                           #
################################################################################

def _plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=history.index, y=history["train_acc"], label="train")
    if "val_acc" in history.columns:
        sns.lineplot(x=history.index, y=history["val_acc"], label="val")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title(f"Learning Curve – {run_id}")
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(path)
    plt.close()
    return path


def _plot_confusion_matrix(cm: np.ndarray, run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {run_id}")
    plt.tight_layout()
    path = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(path)
    plt.close()
    return path

################################################################################
# Main evaluation                                                              #
################################################################################

def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Cross-run evaluation using WandB API")
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help='JSON list e.g. "[\"run-1\", \"run-2\"]"')
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().absolute()
    results_dir.mkdir(parents=True, exist_ok=True)

    run_ids: List[str] = json.loads(args.run_ids)

    # Load global WandB config -------------------------------------------------
    root_cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    root_cfg = OmegaConf.load(root_cfg_path)
    entity, project = root_cfg.wandb.entity, root_cfg.wandb.project

    api = wandb.Api()

    primary_metric = "accuracy"

    aggregated: Dict[str, Dict[str, float]] = {}
    per_run_primary: Dict[str, float] = {}
    generated_paths: List[str] = []
    val_acc_distributions: Dict[str, List[float]] = {}

    # -------------------------------------------------------------------------
    # Per-run processing                                                       
    # -------------------------------------------------------------------------
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        history = run.history()  # pandas DataFrame with full metric history
        summary = dict(run.summary._json_dict)
        cfg = dict(run.config)

        out_dir = results_dir / rid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save run-level metrics ----------------------------------------------
        metrics_path = out_dir / "metrics.json"
        _save_json({"summary": summary, "config": cfg}, metrics_path)
        generated_paths.append(str(metrics_path))

        # Learning curve -------------------------------------------------------
        lc_path = _plot_learning_curve(history, rid, out_dir)
        generated_paths.append(str(lc_path))

        # Confusion matrix -----------------------------------------------------
        if "confusion_matrix" in summary:
            cm = np.array(summary["confusion_matrix"], dtype=int)
            cm_path = _plot_confusion_matrix(cm, rid, out_dir)
            generated_paths.append(str(cm_path))

        # Aggregate scalar metrics --------------------------------------------
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated.setdefault(k, {})[rid] = float(v)
        if "test_acc" in summary:
            per_run_primary[rid] = float(summary["test_acc"])

        # Store val-acc distribution for box plot -----------------------------
        if "val_acc" in history.columns:
            val_acc_distributions[rid] = history["val_acc"].dropna().tolist()

    # -------------------------------------------------------------------------
    # Aggregated comparison                                                    
    # -------------------------------------------------------------------------
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Identify best proposed/baseline runs ------------------------------------
    best_proposed = max(
        ((r, v) for r, v in per_run_primary.items() if "proposed" in r),
        key=lambda x: x[1],
        default=(None, 0.0),
    )
    best_baseline = max(
        ((r, v) for r, v in per_run_primary.items() if any(t in r for t in ("comparative", "baseline"))),
        key=lambda x: x[1],
        default=(None, 0.0),
    )

    # Gap (accuracy is to be maximised) ---------------------------------------
    gap_pct = (best_proposed[1] - best_baseline[1]) / max(best_baseline[1], 1e-9) * 100.0

    # Statistical significance -------------------------------------------------
    proposed_vals = [v for r, v in per_run_primary.items() if "proposed" in r]
    baseline_vals = [v for r, v in per_run_primary.items() if any(t in r for t in ("comparative", "baseline"))]
    p_val = None
    if len(proposed_vals) >= 2 and len(baseline_vals) >= 2:
        _, p_val = ttest_ind(proposed_vals, baseline_vals, equal_var=False)

    agg_metrics = {
        "primary_metric": primary_metric,
        "metrics": aggregated,
        "best_proposed": {"run_id": best_proposed[0], "value": best_proposed[1]},
        "best_baseline": {"run_id": best_baseline[0], "value": best_baseline[1]},
        "gap": gap_pct,
        "p_value": p_val,
    }

    agg_path = comp_dir / "aggregated_metrics.json"
    _save_json(agg_metrics, agg_path)
    generated_paths.append(str(agg_path))

    # Bar chart of primary metric --------------------------------------------
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(per_run_primary.keys()), y=list(per_run_primary.values()))
    plt.ylabel(primary_metric)
    plt.xticks(rotation=45, ha="right")
    plt.title("Test accuracy across runs")
    plt.tight_layout()
    bar_path = comp_dir / "comparison_accuracy_bar_chart.pdf"
    plt.savefig(bar_path)
    plt.close()
    generated_paths.append(str(bar_path))

    # Box plot of validation accuracy distributions ---------------------------
    if val_acc_distributions:
        rows = [(rid, v) for rid, lst in val_acc_distributions.items() for v in lst]
        df_box = pd.DataFrame(rows, columns=["run", "val_acc"])
        plt.figure(figsize=(8, 4))
        sns.boxplot(x="run", y="val_acc", data=df_box)
        plt.ylabel("val_acc")
        plt.xticks(rotation=45, ha="right")
        plt.title("Validation accuracy distribution per run")
        plt.tight_layout()
        box_path = comp_dir / "comparison_val_acc_box_plot.pdf"
        plt.savefig(box_path)
        plt.close()
        generated_paths.append(str(box_path))

    # -------------------------------------------------------------------------
    # Print generated file paths                                               
    # -------------------------------------------------------------------------
    for p in generated_paths:
        print(p)

if __name__ == "__main__":
    main()
