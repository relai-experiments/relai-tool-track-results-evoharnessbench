"""Generate one publication-ready stage-score plot per EOG tool domain.

Run: uv run --locked python plot_results.py
The plots use the pinned website summary in baseline_scores.json and recompute
RELAI stage scores from the copied evaluation reports. No service access needed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analyze import analyze_run, load_json  # noqa: E402


BASELINE_COLORS = {
    "ReAct/Codex deployment": "#6B7280",
    "Raw Memory": "#9A805F",
    "Reasoning Bank": "#7389A4",
    "MemToolAgent": "#608A79",
    "GEPA": "#997A9C",
    "Meta-Harness": "#B17D73",
}
RELAI_COLOR = "#124FD6"
EXPECTED_IDS = {
    "tools-react",
    "tools-raw-memory",
    "tools-reasoning-bank",
    "tools-memtoolagent",
    "tools-gepa",
    "tools-meta-harness",
}


def stage_counts(report: dict, n_stages: int) -> tuple[list[int], list[bool]]:
    counts = []
    terminal_rounded = []
    for stage in range(1, n_stages + 1):
        cells = [row for row in report["matrix"] if int(row["at_stage"]) == stage]
        counts.append(sum(row["n_tasks"] for row in cells))
        terminal_rounded.append(any(row.get("exact") is False for row in cells))
    return counts, terminal_rounded


def validate_baselines(data: dict, manifest: dict) -> None:
    if data.get("schema_version") != 1 or len(data.get("systems", [])) != 6:
        raise ValueError("unexpected baseline data schema or system count")
    if {system["source_id"] for system in data["systems"]} != EXPECTED_IDS:
        raise ValueError("baseline method set changed")
    domains = {entry["domain"] for entry in manifest["runs"]}
    if set(data["domains"]) != domains:
        raise ValueError("baseline and RELAI domain sets differ")
    for system in data["systems"]:
        if system["name"] not in BASELINE_COLORS:
            raise ValueError(f"unrecognized baseline: {system['name']}")
        for entry in manifest["runs"]:
            stage_rows = system["stages"][entry["domain"]]
            if len(stage_rows) != len(entry["stages"]):
                raise ValueError(f"stage count mismatch: {system['name']} / {entry['domain']}")
            for expected_stage, row in enumerate(stage_rows, 1):
                if row["stage"] != expected_stage or row["runs"] != 3:
                    raise ValueError(f"stage or run count mismatch: {system['name']}")
                if not (0 <= row["score"] <= 100 and row["score_sd"] >= 0):
                    raise ValueError(f"invalid baseline score: {system['name']}")


def plot_domain(entry: dict, baselines: dict, output_dir: Path) -> None:
    domain = entry["domain"]
    stage_scores, final_score, _, _ = analyze_run(entry, check=False)
    relai_scores = [100 * value for value in stage_scores]
    relai_scores[-1] = 100 * (stage_scores[-1] + final_score) / 2
    stages = list(range(1, len(relai_scores) + 1))
    report = load_json(ROOT / entry["imported_report"])
    relai_counts, rounded = stage_counts(report, len(stages))
    site_counts = []
    for stage in stages:
        counts = {tuple(system["stages"][domain][stage - 1]["tasks_per_run"]) for system in baselines["systems"]}
        if len(counts) != 1 or len(next(iter(counts))) != 3 or len(set(next(iter(counts)))) != 1:
            raise ValueError(f"inconsistent website task counts: {domain} stage {stage}")
        site_counts.append(next(iter(counts))[0])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    handles = []
    for system in baselines["systems"]:
        rows = system["stages"][domain]
        name = system["name"]
        color = BASELINE_COLORS[name]
        means = [row["score"] for row in rows]
        deviations = [row["score_sd"] for row in rows]
        ax.errorbar(stages, means, yerr=deviations, fmt="none", color=color,
                    alpha=0.43, capsize=2.5, elinewidth=1.0, zorder=2)
        line, = ax.plot(stages, means, label=name, color=color, linewidth=1.6, marker="o",
                        markersize=4.4, alpha=0.86, zorder=3)
        handles.append(line)

    ax.plot(stages, relai_scores, color=RELAI_COLOR, linewidth=3.2, zorder=6)
    for stage, value, is_rounded in zip(stages, relai_scores, rounded):
        ax.scatter(stage, value, s=70, facecolors="white" if is_rounded else RELAI_COLOR,
                   edgecolors=RELAI_COLOR, linewidths=2.3, zorder=7)
    handles.insert(0, Line2D([0], [0], color=RELAI_COLOR, linewidth=3.2,
                             marker="o", markersize=7, label="RELAI"))

    ax.set_title(f"{domain.upper()} · evolving tools", loc="left", fontsize=17, fontweight="bold", pad=17)
    ax.set_xlabel("Harness stage", labelpad=8)
    ax.set_ylabel("Score (%)", labelpad=8)
    ax.set_xticks(stages, [f"v{stage}" for stage in stages])
    ax.set_xlim(0.82, stages[-1] + 0.18)
    ax.set_ylim(30, 80)
    ax.set_yticks(range(30, 81, 10))
    ax.grid(axis="y", color="#DDE2EA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.30)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               bbox_to_anchor=(0.5, 0.125), ncol=4, frameon=False,
               columnspacing=1.4, handlelength=2.2, fontsize=9)
    notes = ["Baseline whiskers: ±1 SD over 3 runs. RELAI final point: mean of matrix and full eval."]
    if any(rounded):
        notes.append("Hollow RELAI markers include terminal-rounded evaluation cells.")
    differences = [(stage, website, local) for stage, website, local in zip(stages, site_counts, relai_counts)
                   if website != local]
    if differences:
        notes.append(f"Website stage {differences[-1][0]}: {differences[-1][1]} tasks; RELAI: {differences[-1][2]} tasks.")
    fig.text(0.5, 0.055, "\n".join(notes), ha="center", va="center", fontsize=8.2,
             linespacing=1.5, color="#4B5563")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": f"{domain.upper()} RELAI vs EvoHarnessBench tool baselines",
                "Description": "Score by cumulative harness stage; RELAI final point averages matrix and full evaluation; baseline error bars show one population standard deviation."}
    svg_path = output_dir / f"{domain}.svg"
    fig.savefig(svg_path, metadata=metadata)
    svg_path.write_text(re.sub(r"[ \t]+(?=\n)", "", svg_path.read_text(encoding="utf-8")),
                        encoding="utf-8")
    fig.savefig(output_dir / f"{domain}.png", dpi=180, metadata=metadata)
    plt.close(fig)
    print(f"{domain}: RELAI {', '.join(f'{value:.2f}' for value in relai_scores)}; "
          f"website tasks {site_counts}; RELAI tasks {relai_counts}")


def main() -> None:
    manifest = load_json(ROOT / "manifest.json")
    baselines = json.loads((ROOT / "baseline_scores.json").read_text(encoding="utf-8"))
    validate_baselines(baselines, manifest)
    for entry in manifest["runs"]:
        plot_domain(entry, baselines, ROOT / "plots")


if __name__ == "__main__":
    main()
