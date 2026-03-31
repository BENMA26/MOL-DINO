#!/usr/bin/env python
import argparse
import csv
import datetime as dt
import os
import subprocess
from typing import Dict, List


def _f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def active_processes() -> List[str]:
    cmd = [
        "bash",
        "-lc",
        (
            "ps -ef | rg "
            "\"run_autoresearch_qm9.sh|trainer_qm9_gen.py|trainer_qm9_ldm.py|run_stage2_distill_then_ldm_every50.sh\" "
            "-n | head -n 12 || true"
        ),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if " rg " in line:
            continue
        lines.append(line[:220])
    return lines


def format_top_rows(rows: List[Dict[str, str]], topk: int = 8) -> str:
    if not rows:
        return "No completed auto-research trials yet.\n"

    sorted_rows = sorted(
        rows,
        key=lambda r: _f(r.get("validity_uniqueness_edm", "")),
        reverse=True,
    )
    lines = [
        "| rank | run_id | atom_stability | molecule_stability | validity | uniqueness | V*U |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, r in enumerate(sorted_rows[:topk], 1):
        lines.append(
            "| "
            f"{idx} | {r.get('run_id', '')} | "
            f"{r.get('atom_stability', '')} | {r.get('molecule_stability', '')} | "
            f"{r.get('validity_edm', '')} | {r.get('uniqueness_edm', '')} | "
            f"{r.get('validity_uniqueness_edm', '')} |"
        )
    return "\n".join(lines) + "\n"


def reached_target(row: Dict[str, str], target_atom: float, target_mol: float, target_valid: float, target_vu: float) -> bool:
    return (
        _f(row.get("atom_stability", "")) >= target_atom
        and _f(row.get("molecule_stability", "")) >= target_mol
        and _f(row.get("validity_edm", "")) >= target_valid
        and _f(row.get("validity_uniqueness_edm", "")) >= target_vu
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_csv", required=True)
    p.add_argument("--log_file", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--latest_note", default="")
    p.add_argument("--target_atom", type=float, default=0.99)
    p.add_argument("--target_mol", type=float, default=0.898)
    p.add_argument("--target_valid", type=float, default=0.95)
    p.add_argument("--target_vu", type=float, default=0.932)
    args = p.parse_args()

    rows = load_rows(args.result_csv)
    rows_sorted = sorted(
        rows,
        key=lambda r: _f(r.get("validity_uniqueness_edm", "")),
        reverse=True,
    )
    best = rows_sorted[0] if rows_sorted else {}
    target_hit = reached_target(
        best, args.target_atom, args.target_mol, args.target_valid, args.target_vu
    ) if best else False
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    proc_lines = active_processes()
    proc_md = "\n".join(f"- `{line}`" for line in proc_lines) if proc_lines else "- None"
    best_line = (
        f"- Best run: `{best.get('run_id', '')}`, "
        f"atom={best.get('atom_stability', '')}, "
        f"mol={best.get('molecule_stability', '')}, "
        f"valid={best.get('validity_edm', '')}, "
        f"V*U={best.get('validity_uniqueness_edm', '')}"
        if best
        else "- Best run: N/A"
    )

    text = f"""# Experiment Log

Last Updated: {now}
Workspace: `{args.workspace}`

## Current Goal

Surpass UniGEM QM9 generation metrics using MOL-DINO based LDM with UNILIP-inspired stage-2 finetuning strategy.

## UniGEM QM9 Target (ICLR 2025)

- Atom stability >= {args.target_atom:.3f} (99.0%)
- Molecule stability >= {args.target_mol:.3f} (89.8%)
- Validity >= {args.target_valid:.3f} (95.0%)
- Validity*Uniqueness >= {args.target_vu:.3f} (93.2%)

## Auto-Research Status

- Result CSV: `{args.result_csv}`
- Completed trials: {len(rows)}
{best_line}
- Target reached: `{"YES" if target_hit else "NO"}`

## Active Processes

{proc_md}

## Top Results (By V*U)

{format_top_rows(rows)}
## Latest Note

- {args.latest_note if args.latest_note else "No additional notes."}
"""

    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
