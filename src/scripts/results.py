import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import src.config as cfg

matplotlib.use("Agg")  # non-interactive backend — saves files without opening windows

FIGURES_DIR    = cfg.MODELS_DIR / "figures"
RESULTS_RPI4   = cfg.MODELS_DIR / "results" / "rpi4"
RESULTS_JETSON = cfg.MODELS_DIR / "results" / "jetson"

# shared entry list used by hardware plots (key, display label)
_BENCH_ENTRIES = [
    ((0.0, "standalone"), "Baseline"),
    ((0.3, "hybrid"),     "p30"),
    ((0.5, "hybrid"),     "p50"),
    ((0.7, "hybrid"),     "p70"),
]

# color scheme: dark = FP32, light = INT8, same hue family per pruning level
_FP32_COLORS = {"Baseline": "#2d2d2d", "p30": "#1f77b4", "p50": "#2ca02c", "p70": "#d62728"}
_INT8_COLORS = {"Baseline": "#888888", "p30": "#7ab8d8", "p50": "#7fcc7f", "p70": "#f08080"}

# known baseline constants as fallback if no report file exists
_BASELINE_FALLBACK = {
    "top1_accuracy":      53.98,
    "theoretical_GFLOPs": 0.528293888,
    "total_parameters_M": 11.269512,
    "physical_size_mb":   43.1,
    "latency_p90_ms":     9.96,
}


# ── data loading ──────────────────────────────────────────────────────────────

# reads all json files from a directory and returns them as a list of dicts
def _load_jsons(directory):
    results = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except Exception:
            continue
    return results


# loads training, pruning and quantization reports from the reports folder
# groups standalone and hybrid reports by (pruning_ratio, stage) and keeps the most recent per key
def load_experiment_data():
    exp = {"baseline": None, "standalone": {}, "hybrid": {}}

    # baseline
    if cfg.REPORTS_BASELINE.exists():
        reports = _load_jsons(cfg.REPORTS_BASELINE)
        if reports:
            latest = max(reports, key=lambda r: r.get("metadata", {}).get("timestamp", ""))
            exp["baseline"] = latest["metrics"]
    # fallback so downstream code always has numbers
    if exp["baseline"] is None:
        exp["baseline"] = _BASELINE_FALLBACK.copy()

    # standalone + hybrid: group by (pruning_ratio, stage), keep most recent
    for workflow, reports_dir in [("standalone", cfg.REPORTS_STANDALONE),
                                  ("hybrid",     cfg.REPORTS_HYBRID)]:
        if not reports_dir.exists():
            continue
        groups = {}
        for report in _load_jsons(reports_dir):
            c     = report.get("config", {})
            ratio = c.get("pruning_ratio")
            stage = c.get("stage")
            if ratio is None or stage is None:
                continue
            key = (ratio, stage)
            ts  = report.get("metadata", {}).get("timestamp", "")
            if key not in groups or ts > groups[key].get("metadata", {}).get("timestamp", ""):
                groups[key] = report
        exp[workflow] = groups

    return exp


# loads edge device benchmark reports and extracts latency, throughput and hardware metrics
# returns a nested dict
# if multiple runs exist for the same model+mode, only the most recent one is kept
def load_benchmark_data(results_dir):
    if not Path(results_dir).exists():
        return {}

    bench = {}
    for report in _load_jsons(results_dir):
        c        = report.get("config", {})
        ratio    = c.get("pruning_ratio", 0.0)
        workflow = c.get("workflow", "standalone")
        mode     = c.get("mode", "fp32")
        ts       = report.get("metadata", {}).get("timestamp", "")

        key = (ratio, workflow)
        if key not in bench:
            bench[key] = {}

        # keep most recent measurement per mode
        if mode not in bench[key] or ts > bench[key][mode].get("_ts", ""):
            bench[key][mode] = {
                "latency_ms": report["single_stream"]["p90_latency_ms"],
                "throughput": report["offline"]["throughput_samples_per_sec"],
                "cpu_pct":    report["hardware"].get("cpu_util_avg_pct"),
                "gpu_pct":    report["hardware"].get("gpu_util_avg_pct"),
                "ram_mb":     report["hardware"].get("process_ram_avg_mb"),
                "temp_max_c": report["hardware"].get("temp_max_c"),
                "size_mb":    report["storage"]["model_size_mb"],
                "_ts":        ts,
            }
    return bench


# ── plot style ────────────────────────────────────────────────────────────────

# global matplotlib style settings applied once before generating all figures
def _apply_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# saves figure as pdf to the figures directory and closes it to free memory
def _save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] saved: {path.name}")


# ── architecture helpers ──────────────────────────────────────────────────────

# standard ResNet-18 output channels per block (used as baseline reference)
_BASELINE_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}


# extracts the highest output channel count per resnet block from the architecture summary
def _get_max_channels_per_block(arch_summary):
    blocks = {"layer1": 0, "layer2": 0, "layer3": 0, "layer4": 0}
    for name, info in arch_summary.items():
        for block in blocks:
            if name.startswith(block) and info["type"] == "conv2d":
                blocks[block] = max(blocks[block], info["C_out"])
    return blocks


# sums up the total flops per resnet block from the architecture summary
def _get_flops_per_block(arch_summary):
    blocks = {"layer1": 0, "layer2": 0, "layer3": 0, "layer4": 0}
    for name, info in arch_summary.items():
        for block in blocks:
            if name.startswith(block):
                blocks[block] += info["flops"]
    return blocks


# finds the standalone finetuned report for a given pruning ratio
def _find_standalone_finetuned(exp, ratio):
    for (r, stage), report in exp["standalone"].items():
        if r == ratio and "finetuned" in stage:
            return report
    return None


# ── 6.1  accuracy recovery ────────────────────────────────────────────────────

# shows how accuracy drops right after pruning and recovers after fine-tuning and quantization
def plot_accuracy_recovery(exp):
    baseline_acc = exp["baseline"]["top1_accuracy"]
    stages       = ["hybrid_raw", "hybrid_finetuned", "hybrid_quantized"]
    stage_labels = ["Raw\nPruned", "Fine-\ntuned", "INT8\nQuantized"]
    ratios       = [0.3, 0.5, 0.7]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(baseline_acc, color="#2d2d2d", lw=1.5,
               label=f"Baseline ({baseline_acc:.1f}%)")

    for i, ratio in enumerate(ratios):
        accs = []
        for stage in stages:
            r = exp["hybrid"].get((ratio, stage))
            accs.append(r["metrics"]["top1_accuracy"] if r else None)

        xs = [j for j, a in enumerate(accs) if a is not None]
        ys = [a for a in accs if a is not None]
        if ys:
            ax.plot(xs, ys,
                    ls="-", marker="o",
                    color=colors[i],
                    label=f"p{int(ratio * 100)}",
                    lw=1.5, ms=6)

    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Accuracy Recovery Through Hybrid Compression Pipeline")
    ax.legend()
    _save(fig, "accuracy_recovery.pdf")


# ── 6.1  compression metrics ──────────────────────────────────────────────────

# shows the relative size reduction from pruning, int8 and the combined effect per pruning level
def plot_compression_metrics(exp):
    b_size = exp["baseline"]["physical_size_mb"]

    ratios = [0.3, 0.5, 0.7]
    labels = ["p30", "p50", "p70"]
    prune_size, int8_size_extra = [], []

    for ratio in ratios:
        r_ft = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        r_q  = exp["hybrid"].get((ratio, "hybrid_quantized"))
        if r_ft:
            size_after_pruning = r_ft["metrics"]["physical_size_mb"]
            prune_size.append((b_size - size_after_pruning) / b_size * 100)
        else:
            prune_size.append(0)
            size_after_pruning = b_size

        if r_q and r_ft:
            size_after_int8 = r_q["metrics"]["physical_size_mb"]
            int8_size_extra.append((size_after_pruning - size_after_int8) / b_size * 100)
        else:
            int8_size_extra.append(0)

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    total  = [p + i for p, i in zip(prune_size, int8_size_extra)]

    x = np.arange(len(labels))
    w = 0.22
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (p, q, t, c) in enumerate(zip(prune_size, int8_size_extra, total, colors)):
        b1 = ax.bar(x[i] - w, p, w, color=c, alpha=0.9)
        b2 = ax.bar(x[i],     q, w, color=c, alpha=0.5, hatch="//")
        b3 = ax.bar(x[i] + w, t, w, color=c, alpha=1.0, edgecolor="black", linewidth=1.2)
        for bar in [b1, b2, b3]:
            h = bar[0].get_height()
            ax.text(bar[0].get_x() + bar[0].get_width() / 2, h + 0.8,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    # legend: color patches for pruning levels + style patches for bar types
    from matplotlib.patches import Patch
    level_handles = [Patch(color=c, label=lbl) for c, lbl in zip(colors, labels)]
    style_handles = [
        Patch(facecolor="0.5", alpha=0.9,                          label="Pruning"),
        Patch(facecolor="0.5", alpha=0.5, hatch="//",              label="INT8"),
        Patch(facecolor="0.5", alpha=1.0, edgecolor="black", linewidth=1.2, label="Combined"),
    ]
    ax.legend(handles=level_handles + style_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reduction vs. Baseline (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Model Size Reduction: Pruning vs. INT8 Contribution")
    _save(fig, "compression_metrics.pdf")


# ── 6.3/6.4  hardware latency ─────────────────────────────────────────────────

# p90 single-stream latency for fp32 and int8
def plot_hardware_latency(bench, device_label, tag):
    labels    = [e[1] for e in _BENCH_ENTRIES]
    fp32_vals = [bench.get(e[0], {}).get("fp32", {}).get("latency_ms", 0) for e in _BENCH_ENTRIES]
    int8_vals = [bench.get(e[0], {}).get("int8", {}).get("latency_ms", 0) for e in _BENCH_ENTRIES]
    colors    = ["#2d2d2d", "#1f77b4", "#2ca02c", "#d62728"]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (fp, i8, c) in enumerate(zip(fp32_vals, int8_vals, colors)):
        ax.bar(x[i] - w/2, fp, w, color=c, alpha=0.9)
        ax.bar(x[i] + w/2, i8, w, color=c, alpha=0.5, hatch="//")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.4", alpha=0.9, label="FP32"),
                       Patch(facecolor="0.4", alpha=0.5, hatch="//", label="INT8")])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P90 Latency (ms)")
    ax.set_title(f"Single-Stream Inference Latency -- {device_label}")
    _save(fig, f"hardware_latency_{tag}.pdf")


# ── throughput ────────────────────────────────────────────────────────────────


# puts raspberry pi and jetson throughput side by side in one figure
def plot_throughput_combined(rpi4_bench, jetson_bench):
    from matplotlib.patches import Patch
    labels  = [e[1] for e in _BENCH_ENTRIES]
    colors  = ["#2d2d2d", "#1f77b4", "#2ca02c", "#d62728"]
    devices = [
        (rpi4_bench,   "Raspberry Pi 4"),
        (jetson_bench, "Jetson Nano"),
    ]

    x = np.arange(len(labels))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, (bench, device_label) in zip(axes, devices):
        fp32_vals = [bench.get(e[0], {}).get("fp32", {}).get("throughput", 0) for e in _BENCH_ENTRIES]
        int8_vals = [bench.get(e[0], {}).get("int8", {}).get("throughput", 0) for e in _BENCH_ENTRIES]
        for i, (fp, i8, c) in enumerate(zip(fp32_vals, int8_vals, colors)):
            ax.bar(x[i] - w/2, fp, w, color=c, alpha=0.9)
            ax.bar(x[i] + w/2, i8, w, color=c, alpha=0.5, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title(f"Offline Throughput -- {device_label}")
        ax.legend(handles=[Patch(facecolor="0.4", alpha=0.9, label="FP32"),
                            Patch(facecolor="0.4", alpha=0.5, hatch="//", label="INT8")])

    _save(fig, "throughput_combined.pdf")


# ── 6.5  theoretical vs real speedup (combined) ──────────────────────────────

# compares theoretical flop reduction against actual latency reduction for both devices in one plot
# the diagonal dashed line shows where theory and reality would match
def plot_theoretical_vs_real(exp, rpi4_bench, jetson_bench):
    b_gflops = exp["baseline"]["theoretical_GFLOPs"]

    devices = [
        (rpi4_bench,   "Raspberry Pi 4", "#1f77b4"),
        (jetson_bench, "Jetson Nano",    "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    all_flop_reds, all_lat_reds = [], []

    for bench, device_label, color in devices:
        b_latency = bench.get((0.0, "standalone"), {}).get("fp32", {}).get("latency_ms")
        if not b_latency:
            print(f"[!] no baseline latency for {device_label} -- skipping")
            continue

        flop_reds, lat_reds, point_labels = [0.0], [0.0], ["Baseline"]
        for ratio in [0.3, 0.5, 0.7]:
            r      = exp["hybrid"].get((ratio, "hybrid_finetuned"))
            b_fp32 = bench.get((ratio, "hybrid"), {}).get("fp32", {})
            if r and b_fp32.get("latency_ms"):
                flop_reds.append((b_gflops - r["metrics"]["theoretical_GFLOPs"]) / b_gflops * 100)
                lat_reds.append((b_latency - b_fp32["latency_ms"]) / b_latency * 100)
                point_labels.append(f"p{int(ratio * 100)}")

        # baseline point black, pruning points in device color
        point_colors = ["black"] + [color] * (len(flop_reds) - 1)
        ax.scatter(flop_reds, lat_reds, color=point_colors, s=70, zorder=5, label=device_label)
        ax.plot(flop_reds, lat_reds, color=color, lw=1.2, alpha=0.7)
        for lbl, x, y in zip(point_labels, flop_reds, lat_reds):
            ax.annotate(lbl, (x, y), xytext=(7, -12), textcoords="offset points", fontsize=8)

        all_flop_reds += flop_reds
        all_lat_reds  += lat_reds

    # ideal line across full data range
    all_vals = all_flop_reds + all_lat_reds
    if all_vals:
        pad = (max(all_vals) - min(all_vals)) * 0.1 or 5
        v_min = min(all_vals) - pad
        v_max = max(all_vals) + pad
        ax.plot([v_min, v_max], [v_min, v_max], "k--", lw=1, alpha=0.4, label="Ideal (linear)")

    ax.set_xlabel("Theoretical FLOP Reduction (%)")
    ax.set_ylabel("Real Latency Reduction (%)")
    ax.set_title("Theoretical vs. Real Speedup -- Raspberry Pi 4 vs. Jetson Nano")
    ax.legend(handles=[
        mlines.Line2D([], [], color="#1f77b4", marker="o", ms=6, lw=1.2, label="Raspberry Pi 4"),
        mlines.Line2D([], [], color="#d62728", marker="o", ms=6, lw=1.2, label="Jetson Nano"),
        mlines.Line2D([], [], color="black", linestyle="--", lw=1, alpha=0.4, label="Ideal (linear)"),
    ])
    _save(fig, "theoretical_vs_real_combined.pdf")


# ── bubble chart ─────────────────────────────────────────────────────────────

# 3d scatter: x=latency, y=accuracy, bubble size=model size in mb
# dark color = fp32, light color = int8 of same pruning level
def plot_bubble_chart(exp, bench, device_label, tag):
    _GROUP_COLORS = {
        "Baseline": "#2d2d2d",
        "p30":      "#1f77b4",
        "p50":      "#2ca02c",
        "p70":      "#d62728",
    }

    from matplotlib.patches import Patch

    b_acc  = exp["baseline"]["top1_accuracy"]
    b_size = exp["baseline"]["physical_size_mb"]
    b_lat  = bench.get((0.0, "standalone"), {}).get("fp32", {}).get("latency_ms")
    points = []
    if b_lat:
        points.append((b_lat, b_acc, b_size, "Baseline", False))
    for ratio in [0.3, 0.5, 0.7]:
        r_ft = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        r_q  = exp["hybrid"].get((ratio, "hybrid_quantized"))
        if not r_ft:
            continue
        acc       = r_ft["metrics"]["top1_accuracy"]
        size_fp32 = r_ft["metrics"]["physical_size_mb"]
        size_int8 = r_q["metrics"]["physical_size_mb"] if r_q else size_fp32
        grp       = f"p{int(ratio * 100)}"
        lat_fp32  = bench.get((ratio, "hybrid"), {}).get("fp32", {}).get("latency_ms")
        lat_int8  = bench.get((ratio, "hybrid"), {}).get("int8", {}).get("latency_ms")
        if lat_fp32:
            points.append((lat_fp32, acc, size_fp32, grp, False))
        if lat_int8:
            points.append((lat_int8, acc, size_int8, grp, True))

    if not points:
        print(f"[!] no data for bubble chart ({device_label})")
        return

    max_size = max(p[2] for p in points)
    fig, ax  = plt.subplots(figsize=(8, 5))
    for lat, acc, size_mb, grp, is_int8 in points:
        color = _INT8_COLORS[grp] if is_int8 else _FP32_COLORS[grp]
        area  = (size_mb / max_size) * 800
        ax.scatter(lat, acc, s=area, color=color, alpha=0.85, zorder=4)

    ax.margins(x=0.15, y=0.15)
    ax.set_xlabel("P90 Latency (ms)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Accuracy * Latency * Model Size -- {device_label}")

    group_handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                     markerfacecolor=_FP32_COLORS[grp], markersize=8, label=grp)
                     for grp in _FP32_COLORS if any(p[3] == grp for p in points)]
    style_handles = [
        Patch(facecolor="0.3", label="FP32 (dark)"),
        Patch(facecolor="0.7", label="INT8 (light)"),
    ]
    ax.legend(handles=group_handles + style_handles, fontsize=8, loc="lower right", framealpha=0.85)
    _save(fig, f"bubble_chart_{tag}.pdf")


# ── resource utilization ──────────────────────────────────────────────────────

def plot_resource_utilization(bench, device_label, tag):
    from matplotlib.patches import Patch

    labels   = [e[1] for e in _BENCH_ENTRIES]
    colors   = ["#2d2d2d", "#1f77b4", "#2ca02c", "#d62728"]
    cpu_fp32 = [bench.get(e[0], {}).get("fp32", {}).get("cpu_pct", 0) or 0 for e in _BENCH_ENTRIES]
    cpu_int8 = [bench.get(e[0], {}).get("int8", {}).get("cpu_pct", 0) or 0 for e in _BENCH_ENTRIES]
    gpu_fp32 = [bench.get(e[0], {}).get("fp32", {}).get("gpu_pct", 0) or 0 for e in _BENCH_ENTRIES]
    gpu_int8 = [bench.get(e[0], {}).get("int8", {}).get("gpu_pct", 0) or 0 for e in _BENCH_ENTRIES]
    ram_fp32 = [bench.get(e[0], {}).get("fp32", {}).get("ram_mb",  0) or 0 for e in _BENCH_ENTRIES]
    ram_int8 = [bench.get(e[0], {}).get("int8", {}).get("ram_mb",  0) or 0 for e in _BENCH_ENTRIES]

    has_gpu = any(gpu_fp32 + gpu_int8)
    ncols   = 3 if has_gpu else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
    if ncols == 2:
        ax_cpu, ax_ram = axes
        ax_gpu = None
    else:
        ax_cpu, ax_gpu, ax_ram = axes

    x = np.arange(len(labels))
    w = 0.35

    for i, (cf, ci, c) in enumerate(zip(cpu_fp32, cpu_int8, colors)):
        ax_cpu.bar(x[i] - w/2, cf, w, color=c, alpha=0.9)
        ax_cpu.bar(x[i] + w/2, ci, w, color=c, alpha=0.5, hatch="//")

    ax_cpu.set_xticks(x); ax_cpu.set_xticklabels(labels)
    ax_cpu.set_ylabel("CPU Utilization (%)")
    cpu_max = max(cpu_fp32 + cpu_int8) if any(cpu_fp32 + cpu_int8) else 10
    ax_cpu.set_ylim(0, min(cpu_max * 1.2, 110))
    ax_cpu.set_title(f"CPU Utilization -- {device_label}")

    if ax_gpu is not None:
        for i, (gf, gi, c) in enumerate(zip(gpu_fp32, gpu_int8, colors)):
            ax_gpu.bar(x[i] - w/2, gf, w, color=c, alpha=0.9)
            ax_gpu.bar(x[i] + w/2, gi, w, color=c, alpha=0.5, hatch="//")

        ax_gpu.set_xticks(x); ax_gpu.set_xticklabels(labels)
        ax_gpu.set_ylabel("GPU Utilization (%)")
        gpu_max = max(gpu_fp32 + gpu_int8) if any(gpu_fp32 + gpu_int8) else 10
        ax_gpu.set_ylim(0, min(gpu_max * 1.2, 110))
        ax_gpu.set_title(f"GPU Utilization -- {device_label}")

    for i, (rf, ri, c) in enumerate(zip(ram_fp32, ram_int8, colors)):
        ax_ram.bar(x[i] - w/2, rf, w, color=c, alpha=0.9)
        ax_ram.bar(x[i] + w/2, ri, w, color=c, alpha=0.5, hatch="//")

    ax_ram.set_xticks(x); ax_ram.set_xticklabels(labels)
    ax_ram.set_ylabel("Process RAM (MB)")
    ax_ram.set_title(f"RAM Utilization -- {device_label}")

    legend_handles = [Patch(facecolor="0.4", alpha=0.9, label="FP32"),
                      Patch(facecolor="0.4", alpha=0.5, hatch="//", label="INT8")]
    fig.legend(handles=legend_handles, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=2, framealpha=0.85)
    _save(fig, f"resource_utilization_{tag}.pdf")


# ── 6.7  pareto frontier ──────────────────────────────────────────────────────

# latency and throughput pareto plots side by side in one figure
def plot_pareto_combined(exp, bench, device_label, tag):
    _GROUP_COLORS = {
        "Baseline": "#2d2d2d",
        "p30":      "#1f77b4",
        "p50":      "#2ca02c",
        "p70":      "#d62728",
    }

    b_acc = exp["baseline"]["top1_accuracy"]

    # collects latency points
    lat_points = []
    b_lat = bench.get((0.0, "standalone"), {}).get("fp32", {}).get("latency_ms")
    if b_lat:
        lat_points.append((b_lat, b_acc, "Baseline", False))

    for ratio in [0.3, 0.5, 0.7]:
        r = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        if not r:
            continue
        acc      = r["metrics"]["top1_accuracy"]
        grp      = f"p{int(ratio * 100)}"
        lat_fp32 = bench.get((ratio, "hybrid"), {}).get("fp32", {}).get("latency_ms")
        lat_int8 = bench.get((ratio, "hybrid"), {}).get("int8", {}).get("latency_ms")
        if lat_fp32:
            lat_points.append((lat_fp32, acc, grp, False))
        if lat_int8:
            lat_points.append((lat_int8, acc, grp, True))

    # collect throughput points
    thr_points = []
    b_thr = bench.get((0.0, "standalone"), {}).get("fp32", {}).get("throughput")
    if b_thr:
        thr_points.append((b_thr, b_acc, "Baseline", False))

    for ratio in [0.3, 0.5, 0.7]:
        r = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        if not r:
            continue
        acc      = r["metrics"]["top1_accuracy"]
        grp      = f"p{int(ratio * 100)}"
        thr_fp32 = bench.get((ratio, "hybrid"), {}).get("fp32", {}).get("throughput")
        thr_int8 = bench.get((ratio, "hybrid"), {}).get("int8", {}).get("throughput")
        if thr_fp32:
            thr_points.append((thr_fp32, acc, grp, False))
        if thr_int8:
            thr_points.append((thr_int8, acc, grp, True))

    def pareto_latency(points):
        return sorted([p for p in points if not any(
            o[0] <= p[0] and o[1] >= p[1] and (o[0] < p[0] or o[1] > p[1])
            for o in points
        )], key=lambda p: p[0])

    def pareto_throughput(points):
        return sorted([p for p in points if not any(
            o[0] >= p[0] and o[1] >= p[1] and (o[0] > p[0] or o[1] > p[1])
            for o in points
        )], key=lambda p: p[0])

    style_handles = [
        mlines.Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor="0.3", markersize=8, label="FP32 (dark)"),
        mlines.Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor="0.7", markersize=8, label="INT8 (light)"),
        mlines.Line2D([], [], linestyle="--", color="black",
                      lw=1.5, label="Pareto Frontier"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # latency subplot
    for lat, acc, grp, is_int8 in lat_points:
        color = _INT8_COLORS[grp] if is_int8 else _FP32_COLORS[grp]
        ax1.scatter(lat, acc, s=60, color=color, zorder=4)
    pareto_l = pareto_latency(lat_points)

    if len(pareto_l) > 1:
        ax1.plot([p[0] for p in pareto_l], [p[1] for p in pareto_l], "k--", lw=1.5, zorder=3)
    ax1.set_xlabel("P90 Latency (ms)")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title(f"Pareto: Accuracy vs. Latency -- {device_label}")

    # throughput subplot
    for thr, acc, grp, is_int8 in thr_points:
        color = _INT8_COLORS[grp] if is_int8 else _FP32_COLORS[grp]
        ax2.scatter(thr, acc, s=60, color=color, zorder=4)
    pareto_t = pareto_throughput(thr_points)

    if len(pareto_t) > 1:
        ax2.plot([p[0] for p in pareto_t], [p[1] for p in pareto_t], "k--", lw=1.5, zorder=3)
    ax2.set_xlabel("Throughput (samples/sec)")
    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title(f"Pareto: Accuracy vs. Throughput -- {device_label}")

    # shared legend below both subplots
    group_handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                     markerfacecolor=_FP32_COLORS[grp], markersize=8, label=grp)
                     for grp in _FP32_COLORS if any(p[2] == grp for p in lat_points)]
    fig.legend(handles=group_handles + style_handles, fontsize=8, ncol=len(group_handles) + len(style_handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.08), framealpha=0.85)

    _save(fig, f"pareto_combined_{tag}.pdf")


# ── summary table ─────────────────────────────────────────────────────────────

# prints a formatted overview of all key metrics for both devices to the terminal
def print_summary_table(exp, rpi4_results, jetson_results):
    b      = exp["baseline"]
    b_acc  = b["top1_accuracy"]
    b_rpi4 = rpi4_results.get((0.0, "standalone"), {})
    b_jet  = jetson_results.get((0.0, "standalone"), {}) if jetson_results else {}
    b_lat_rpi4 = b_rpi4.get("fp32", {}).get("latency_ms", 0)
    b_lat_jet  = b_jet.get("fp32",  {}).get("latency_ms", 0)

    def d_acc(acc):
        return f"{acc - b_acc:+.2f}" if acc else "—"

    def d_lat(lat, b_lat):
        if not lat or not b_lat:
            return "—"
        return f"{(lat - b_lat) / b_lat * 100:+.1f}%"

    cols = 112
    print("\n" + "=" * cols)
    print("RESULTS SUMMARY TABLE")
    print("=" * cols)
    print(f"{'Model':<22} {'Acc%':>6} {'ΔAcc':>6} {'GFLOPs':>7} {'Params M':>9} "
          f"{'Size MB':>8} {'Pi4 ms':>8} {'ΔPi4':>7} {'Jet ms':>8} {'ΔJet':>7}")
    print("-" * cols)

    # baseline row
    print(f"{'Baseline':<22} {b_acc:>6.2f} {'—':>6} {b['theoretical_GFLOPs']:>7.3f} "
          f"{b['total_parameters_M']:>9.2f} {b['physical_size_mb']:>8.2f} "
          f"{b_lat_rpi4:>8.2f} {'-':>7} {b_lat_jet:>8.2f} {'-':>7}")

    for ratio in [0.3, 0.5, 0.7]:
        r_ft = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        r_q  = exp["hybrid"].get((ratio, "hybrid_quantized"))
        rpi4 = rpi4_results.get((ratio, "hybrid"), {})
        jet  = jetson_results.get((ratio, "hybrid"), {}) if jetson_results else {}
        if not r_ft:
            continue

        m_ft   = r_ft["metrics"]
        lbl    = f"p{int(ratio * 100)}"
        gflops = m_ft["theoretical_GFLOPs"]
        params = m_ft["total_parameters_M"]

        # FP32 row
        acc_fp32      = m_ft["top1_accuracy"]
        size_fp32     = m_ft["physical_size_mb"]
        lat_rpi4_fp32 = rpi4.get("fp32", {}).get("latency_ms", 0)
        lat_jet_fp32  = jet.get("fp32",  {}).get("latency_ms", 0)

        print(f"{lbl + ' FP32':<22} {acc_fp32:>6.2f} {d_acc(acc_fp32):>6} "
              f"{gflops:>7.3f} {params:>9.2f} {size_fp32:>8.2f} "
              f"{lat_rpi4_fp32:>8.2f} {d_lat(lat_rpi4_fp32, b_lat_rpi4):>7} "
              f"{lat_jet_fp32:>8.2f} {d_lat(lat_jet_fp32, b_lat_jet):>7}")

        # INT8 row (GFLOPs/Params unchanged by quantization → show dashes)
        acc_int8      = r_q["metrics"]["top1_accuracy"]  if r_q else acc_fp32
        size_int8     = r_q["metrics"]["physical_size_mb"] if r_q else size_fp32
        lat_rpi4_int8 = rpi4.get("int8", {}).get("latency_ms", 0)
        lat_jet_int8  = jet.get("int8",  {}).get("latency_ms", 0)

        print(f"{lbl + ' INT8':<22} {acc_int8:>6.2f} {d_acc(acc_int8):>6} "
              f"{'-':>7} {'-':>9} {size_int8:>8.2f} "
              f"{lat_rpi4_int8:>8.2f} {d_lat(lat_rpi4_int8, b_lat_rpi4):>7} "
              f"{lat_jet_int8:>8.2f} {d_lat(lat_jet_int8, b_lat_jet):>7}")

    print("=" * cols)


# ── channel reduction & flops per layer block (combined) ─────────────────────

# shows how channel count and mflops change per resnet block after pruning
# baseline channels are hardcoded since there is no separate baseline architecture report
def plot_channel_and_flops(exp):
    blocks = ["layer1", "layer2", "layer3", "layer4"]
    labels = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]
    ratios = [0.3, 0.5, 0.7]
    colors = ["#2d2d2d", "#1f77b4", "#2ca02c", "#d62728"]
    x      = range(len(labels))

    # channel data
    channel_data = {"Baseline": [_BASELINE_CHANNELS[b] for b in blocks]}
    for ratio in ratios:
        r = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        if r and "architecture_summary" in r:
            ch = _get_max_channels_per_block(r["architecture_summary"])
            channel_data[f"p{int(ratio * 100)}"] = [ch[b] for b in blocks]

    # flops data
    flops_data = {}
    for (ratio, stage), report in exp["standalone"].items():
        if ratio == 0.0 and "architecture_summary" in report:
            flops = _get_flops_per_block(report["architecture_summary"])
            flops_data["Baseline"] = [flops[b] / 1e6 for b in blocks]
            break

    for ratio in ratios:
        r = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        if r and "architecture_summary" in r:
            flops = _get_flops_per_block(r["architecture_summary"])
            flops_data[f"p{int(ratio * 100)}"] = [flops[b] / 1e6 for b in blocks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, (name, vals) in enumerate(channel_data.items()):
        ax1.plot(x, vals, ls="-", marker="o", color=colors[i], lw=1.5, ms=6, label=name)

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Max Output Channels")
    ax1.set_title("Channel Count per ResNet Block")
    ax1.legend()

    for i, (name, vals) in enumerate(flops_data.items()):
        ax2.plot(x, vals, ls="-", marker="o", color=colors[i], lw=1.5, ms=6, label=name)

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("MFLOPs")
    ax2.set_title("Computational Cost per ResNet Block")
    ax2.legend()

    fig.suptitle("Structural Impact of Pruning across ResNet Blocks", fontsize=11)
    _save(fig, "channel_and_flops_per_layer.pdf")


# ── accuracy vs gflops ────────────────────────────────────────────────────────

# scatter plot showing the accuracy-flops tradeoff across all pruning levels
# uses theoretical gflops from the reports, not measured runtime flops
def plot_accuracy_vs_gflops(exp):
    colors = {"p30": "#1f77b4", "p50": "#2ca02c", "p70": "#d62728"}

    fig, ax = plt.subplots(figsize=(7, 5))

    b = exp["baseline"]
    gflops_seq = [b["theoretical_GFLOPs"]]
    acc_seq    = [b["top1_accuracy"]]

    ax.scatter(b["theoretical_GFLOPs"], b["top1_accuracy"],
               color="black", s=80, zorder=5)
    ax.annotate("Baseline", (b["theoretical_GFLOPs"], b["top1_accuracy"]),
                xytext=(6, -14), textcoords="offset points", fontsize=8)

    label_offsets = {0.3: (6, -14), 0.5: (6, -14), 0.7: (6, 6)}
    for ratio in [0.3, 0.5, 0.7]:
        r = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        if r:
            m   = r["metrics"]
            grp = f"p{int(ratio * 100)}"
            ax.scatter(m["theoretical_GFLOPs"], m["top1_accuracy"],
                       color=colors[grp], s=80, zorder=5)
            ax.annotate(grp, (m["theoretical_GFLOPs"], m["top1_accuracy"]),
                        xytext=label_offsets[ratio], textcoords="offset points", fontsize=8)
            gflops_seq.append(m["theoretical_GFLOPs"])
            acc_seq.append(m["top1_accuracy"])

    # connect points with a thin gray dashed line
    ax.plot(gflops_seq, acc_seq, color="0.6", lw=1, linestyle="--", zorder=3)

    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs. Computational Cost")
    ax.set_xlim(left=-0.01, right=max(gflops_seq) * 1.18)
    _save(fig, "accuracy_vs_gflops.pdf")


# ── training time ─────────────────────────────────────────────────────────────

# shows how long fine-tuning took for standalone vs hybrid at each pruning level
def plot_training_time(exp):
    ratios = [0.3, 0.5, 0.7]
    labels = ["p30", "p50", "p70"]

    standalone_times, hybrid_times = [], []
    for ratio in ratios:
        r_s = _find_standalone_finetuned(exp, ratio)
        r_h = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        standalone_times.append(r_s["metrics"]["total_training_time_sec"] / 60 if r_s else 0)
        hybrid_times.append(r_h["metrics"]["total_training_time_sec"] / 60 if r_h else 0)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, standalone_times, w, label="Standalone", color="0.4")
    ax.bar(x + w / 2, hybrid_times,     w, label="Hybrid",     color="0.7", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fine-Tuning Time (minutes)")
    ax.set_title("Fine-Tuning Training Time per Pruning Configuration")
    ax.legend()
    _save(fig, "training_time.pdf")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    print("[*] loading experiment reports...")
    experiment_data = load_experiment_data()

    print("[*] loading Raspberry Pi 4 benchmark reports...")
    rpi4_results = load_benchmark_data(RESULTS_RPI4)

    print("[*] loading Jetson Nano benchmark reports...")
    jetson_results = load_benchmark_data(RESULTS_JETSON)

    print("\n[*] generating figures...")

    # plots that only need experiment data
    plot_accuracy_recovery(experiment_data)
    plot_compression_metrics(experiment_data)
    plot_channel_and_flops(experiment_data)
    plot_accuracy_vs_gflops(experiment_data)
    plot_training_time(experiment_data)

    # plots that need hardware benchmark data
    for bench_data, device_label, device_tag in [
        (rpi4_results,   "Raspberry Pi 4", "rpi4"),
        (jetson_results, "Jetson Nano",    "jetson"),
    ]:
        if not bench_data:
            print(f"[!] no benchmark data for {device_label} -- skipping hardware plots")
            continue
        plot_hardware_latency(bench_data, device_label, device_tag)
        plot_bubble_chart(experiment_data, bench_data, device_label, device_tag)
        plot_resource_utilization(bench_data, device_label, device_tag)
        plot_pareto_combined(experiment_data, bench_data, device_label, device_tag)

    if rpi4_results and jetson_results:
        plot_throughput_combined(rpi4_results, jetson_results)
        plot_theoretical_vs_real(experiment_data, rpi4_results, jetson_results)

    print_summary_table(experiment_data, rpi4_results, jetson_results)
    print(f"\n[!] all figures saved to: {FIGURES_DIR}")
