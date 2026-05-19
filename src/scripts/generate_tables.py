import matplotlib
import matplotlib.pyplot as plt
import src.config as cfg
from src.scripts.results import (
    load_experiment_data,
    load_benchmark_data,
    RESULTS_RPI4,
    RESULTS_JETSON,
    _BENCH_ENTRIES,
)

matplotlib.use("Agg")

TABLES_DIR = cfg.MODELS_DIR / "tables"

# saves table figure as pdf to the tables directory
def _save_table(fig, name):
    path = TABLES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] saved: {path.name}")

# col_widths controls relative column sizes, defaults to equal distribution
def _render_table(headers, rows, col_widths=None, title=None):
    n_cols     = len(headers)
    n_rows     = len(rows)
    # figure height scales with number of rows
    fig_h      = max(1.0, 0.3 * (n_rows + 2))
    col_widths = col_widths or [1 / n_cols] * n_cols

    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=9, pad=6)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.3)

    # light gray header row
    for j in range(n_cols):
        table[0, j].set_facecolor("#e0e0e0")

    return fig

# ── accuracy recovery ─────────────────────────────────────────────────────────

# shows top-1 accuracy at each stage of the hybrid pipeline for all pruning levels
def table_accuracy_recovery(exp):
    baseline = exp["baseline"]["top1_accuracy"]
    headers  = ["Configuration", "Raw Pruned (%)", "Fine-tuned (%)", "INT8 Quantized (%)"]
    # baseline has no raw or quantized stage so those cells show a dash
    rows     = [["Baseline", "-", f"{baseline:.2f}", "-"]]

    for ratio in [0.3, 0.5, 0.7]:
        tag = f"p{int(ratio * 100)}"
        raw = exp["hybrid"].get((ratio, "hybrid_raw"))
        ft  = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        qt  = exp["hybrid"].get((ratio, "hybrid_quantized"))
        rows.append([
            tag,
            f"{raw['metrics']['top1_accuracy']:.2f}" if raw else "-",
            f"{ft['metrics']['top1_accuracy']:.2f}"  if ft  else "-",
            f"{qt['metrics']['top1_accuracy']:.2f}"  if qt  else "-",
        ])

    fig = _render_table(headers, rows, title="Accuracy Recovery Through Hybrid Compression Pipeline")
    _save_table(fig, "table_accuracy_recovery.pdf")

# ── compression metrics ───────────────────────────────────────────────────────

# breaks down size reduction into pruning and int8 contributions per pruning level
def table_compression_metrics(exp):
    b_size  = exp["baseline"]["physical_size_mb"]
    headers = ["Config", "Baseline (MB)", "After Pruning (MB)", "After INT8 (MB)",
               "Pruning Red. (%)", "INT8 Red. (%)", "Total Red. (%)"]
    rows    = []

    for ratio in [0.3, 0.5, 0.7]:
        tag = f"p{int(ratio * 100)}"
        ft  = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        qt  = exp["hybrid"].get((ratio, "hybrid_quantized"))
        if not ft:
            continue
        s_ft  = ft["metrics"]["physical_size_mb"]
        s_qt  = qt["metrics"]["physical_size_mb"] if qt else s_ft
        # all reductions are relative to the original baseline size
        p_red = (b_size - s_ft) / b_size * 100
        i_red = (s_ft  - s_qt) / b_size * 100
        t_red = (b_size - s_qt) / b_size * 100
        rows.append([tag, f"{b_size:.2f}", f"{s_ft:.2f}", f"{s_qt:.2f}",
                     f"{p_red:.1f}", f"{i_red:.1f}", f"{t_red:.1f}"])

    fig = _render_table(headers, rows, title="Model Size Reduction: Pruning vs. INT8 Contribution")
    _save_table(fig, "table_compression_metrics.pdf")


# ── hardware latency ──────────────────────────────────────────────────────────

# p90 single-stream latency for fp32 and int8 on the raspberry pi
def table_hardware_latency_rpi4(rpi4):
    headers = ["Model", "FP32 (ms)", "INT8 (ms)"]
    rows    = []

    # _BENCH_ENTRIES covers baseline + p30/p50/p70 hybrid
    for key, label in _BENCH_ENTRIES:
        fp = rpi4.get(key, {}).get("fp32", {}).get("latency_ms")
        i8 = rpi4.get(key, {}).get("int8", {}).get("latency_ms")
        rows.append([label,
                     f"{fp:.2f}" if fp else "-",
                     f"{i8:.2f}" if i8 else "-"])

    fig = _render_table(headers, rows, title="Single-Stream P90 Latency (ms) - Raspberry Pi 4")
    _save_table(fig, "table_hardware_latency_rpi4.pdf")

# p90 single-stream latency for fp32 and int8 on the jetson nano
def table_hardware_latency_jetson(jetson):
    headers = ["Model", "FP32 (ms)", "INT8 (ms)"]
    rows    = []

    for key, label in _BENCH_ENTRIES:
        fp = jetson.get(key, {}).get("fp32", {}).get("latency_ms")
        i8 = jetson.get(key, {}).get("int8", {}).get("latency_ms")
        rows.append([label,
                     f"{fp:.2f}" if fp else "-",
                     f"{i8:.2f}" if i8 else "-"])

    fig = _render_table(headers, rows, title="Single-Stream P90 Latency (ms) - Jetson Nano")
    _save_table(fig, "table_hardware_latency_jetson.pdf")


# ── throughput ────────────────────────────────────────────────────────────────

# offline throughput for both devices in one table for easy comparison
def table_throughput(rpi4, jetson):
    headers = ["Model", "RPi4 FP32 (s/s)", "RPi4 INT8 (s/s)", "Jetson FP32 (s/s)", "Jetson INT8 (s/s)"]
    rows    = []

    for key, label in _BENCH_ENTRIES:
        r4_fp = rpi4.get(key, {}).get("fp32", {}).get("throughput")
        r4_i8 = rpi4.get(key, {}).get("int8", {}).get("throughput")
        jt_fp = jetson.get(key, {}).get("fp32", {}).get("throughput")
        jt_i8 = jetson.get(key, {}).get("int8", {}).get("throughput")
        rows.append([label,
                     f"{r4_fp:.1f}" if r4_fp else "-",
                     f"{r4_i8:.1f}" if r4_i8 else "-",
                     f"{jt_fp:.1f}" if jt_fp else "-",
                     f"{jt_i8:.1f}" if jt_i8 else "-"])

    fig = _render_table(headers, rows, title="Offline Throughput (samples/sec)")
    _save_table(fig, "table_throughput.pdf")


# ── resource utilization ──────────────────────────────────────────────────────

# cpu and ram usage during inference on raspberry pi
def table_resource_rpi4(rpi4):
    headers = ["Model", "CPU FP32 (%)", "CPU INT8 (%)", "RAM FP32 (MB)", "RAM INT8 (MB)"]
    rows    = []

    for key, label in _BENCH_ENTRIES:
        fp = rpi4.get(key, {}).get("fp32", {})
        i8 = rpi4.get(key, {}).get("int8", {})
        rows.append([label,
                     f"{fp.get('cpu_pct', 0):.1f}" if fp.get("cpu_pct") else "-",
                     f"{i8.get('cpu_pct', 0):.1f}" if i8.get("cpu_pct") else "-",
                     f"{fp.get('ram_mb',  0):.1f}" if fp.get("ram_mb")  else "-",
                     f"{i8.get('ram_mb',  0):.1f}" if i8.get("ram_mb")  else "-"])

    fig = _render_table(headers, rows, title="Resource Utilization - Raspberry Pi 4")
    _save_table(fig, "table_resource_rpi4.pdf")


# cpu, gpu and ram usage during inference on the jetson nano
# gpu column only has values if the device reported gpu data
def table_resource_jetson(jetson):
    headers = ["Model", "CPU FP32 (%)", "CPU INT8 (%)", "GPU FP32 (%)", "GPU INT8 (%)",
               "RAM FP32 (MB)", "RAM INT8 (MB)"]
    rows    = []

    for key, label in _BENCH_ENTRIES:
        fp = jetson.get(key, {}).get("fp32", {})
        i8 = jetson.get(key, {}).get("int8", {})
        rows.append([label,
                     f"{fp.get('cpu_pct', 0):.1f}" if fp.get("cpu_pct") else "-",
                     f"{i8.get('cpu_pct', 0):.1f}" if i8.get("cpu_pct") else "-",
                     f"{fp.get('gpu_pct', 0):.1f}" if fp.get("gpu_pct") else "-",
                     f"{i8.get('gpu_pct', 0):.1f}" if i8.get("gpu_pct") else "-",
                     f"{fp.get('ram_mb',  0):.1f}" if fp.get("ram_mb")  else "-",
                     f"{i8.get('ram_mb',  0):.1f}" if i8.get("ram_mb")  else "-"])

    # narrower columns needed because of 7 columns
    fig = _render_table(headers, rows,
                        col_widths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.14, 0.14],
                        title="Resource Utilization - Jetson Nano")
    _save_table(fig, "table_resource_jetson.pdf")


# ── theoretical vs real ───────────────────────────────────────────────────────

# compares theoretical flop reduction against actual latency reduction for both devices
def table_theoretical_vs_real(exp, rpi4, jetson):
    b_gflops  = exp["baseline"]["theoretical_GFLOPs"]
    # baseline latency needed to calculate percentage reduction for each pruning level
    b_lat_rpi = rpi4.get((0.0, "standalone"), {}).get("fp32", {}).get("latency_ms")
    b_lat_jet = jetson.get((0.0, "standalone"), {}).get("fp32", {}).get("latency_ms")

    headers = ["Config", "FLOP Red. (%)", "RPi4 Real Red. (%)", "Jetson Real Red. (%)"]
    rows    = []

    for ratio in [0.3, 0.5, 0.7]:
        r      = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        rpi_fp = rpi4.get((ratio, "hybrid"), {}).get("fp32", {}).get("latency_ms")
        jet_fp = jetson.get((ratio, "hybrid"), {}).get("fp32", {}).get("latency_ms")
        if not r:
            continue
        flop_red = (b_gflops - r["metrics"]["theoretical_GFLOPs"]) / b_gflops * 100
        rpi_red  = (b_lat_rpi - rpi_fp) / b_lat_rpi * 100 if b_lat_rpi and rpi_fp else None
        jet_red  = (b_lat_jet - jet_fp) / b_lat_jet * 100 if b_lat_jet and jet_fp else None
        rows.append([f"p{int(ratio * 100)}",
                     f"{flop_red:.1f}",
                     f"{rpi_red:.1f}" if rpi_red else "-",
                     f"{jet_red:.1f}" if jet_red else "-"])

    fig = _render_table(headers, rows, title="Theoretical vs. Real Speedup")
    _save_table(fig, "table_theoretical_vs_real.pdf")


# ── accuracy vs gflops ────────────────────────────────────────────────────────

# lists gflops, parameter count and accuracy for baseline and all pruning levels
def table_accuracy_vs_gflops(exp):
    b            = exp["baseline"]
    baseline_acc = b["top1_accuracy"]
    headers      = ["Config", "GFLOPs", "Parameters (M)", "Top-1 Acc. (%)", "Acc. Drop (pp)"]
    # baseline has no accuracy drop so that cell shows a dash
    rows         = [["Baseline", f"{b['theoretical_GFLOPs']:.3f}",
                     f"{b['total_parameters_M']:.2f}", f"{baseline_acc:.2f}", "-"]]

    for ratio in [0.3, 0.5, 0.7]:
        r = exp["hybrid"].get((ratio, "hybrid_finetuned"))
        if not r:
            continue
        m    = r["metrics"]
        drop = m["top1_accuracy"] - baseline_acc
        rows.append([f"p{int(ratio * 100)}",
                     f"{m['theoretical_GFLOPs']:.3f}",
                     f"{m['total_parameters_M']:.2f}",
                     f"{m['top1_accuracy']:.2f}",
                     f"{drop:+.2f}"])

    fig = _render_table(headers, rows, title="Accuracy vs. Computational Cost")
    _save_table(fig, "table_accuracy_vs_gflops.pdf")

if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] loading data...")
    exp    = load_experiment_data()
    rpi4   = load_benchmark_data(RESULTS_RPI4)
    jetson = load_benchmark_data(RESULTS_JETSON)

    print("[*] generating tables...")
    table_accuracy_recovery(exp)
    table_compression_metrics(exp)
    table_hardware_latency_rpi4(rpi4)
    table_hardware_latency_jetson(jetson)
    table_throughput(rpi4, jetson)
    table_resource_rpi4(rpi4)
    table_resource_jetson(jetson)
    table_theoretical_vs_real(exp, rpi4, jetson)
    table_accuracy_vs_gflops(exp)

    print(f"\nall tables saved to: {TABLES_DIR}")
