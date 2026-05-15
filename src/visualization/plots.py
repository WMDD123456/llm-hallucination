"""实验结果可视化"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 尝试使用中文字体
_font_found = False
for font_name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
    for f in fm.fontManager.ttflist:
        if font_name in f.name:
            plt.rcParams["font.sans-serif"] = [f.name]
            _font_found = True
            break
    if _font_found:
        break

if not _font_found:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print("警告：未找到中文字体，图表中文可能显示为方框。")

plt.rcParams["axes.unicode_minus"] = False


def plot_hallucination_rate(metrics: dict, save_path: str | Path = "data/results/hr_comparison.png"):
    """幻觉率对比柱状图"""
    models = list(metrics.keys())
    groups = ["G1", "G2", "G3", "G4", "G5"]
    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        hr_values = [metrics[model].get(g, {}).get("HR", 0) for g in groups]
        bars = ax.bar(x + i * width, hr_values, width, label=model)

        for bar, val in zip(bars, hr_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("实验组")
    ax.set_ylabel("幻觉率 (%)")
    ax.set_title("各实验组幻觉率对比")
    ax.set_xticks(x + width)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"幻觉率对比图已保存至: {save_path}")


def plot_mitigation_gain(metrics: dict, save_path: str | Path = "data/results/mg_comparison.png"):
    """缓解增益对比图"""
    from src.evaluation.metrics import compute_mitigation_gain
    gains = compute_mitigation_gain(metrics)

    models = list(gains.keys())
    groups = ["G2", "G3", "G4", "G5"]
    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        mg_values = [gains[model].get(g, 0) for g in groups]
        bars = ax.bar(x + i * width, mg_values, width, label=model)

        for bar, val in zip(bars, mg_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("实验组（相对于 G1）")
    ax.set_ylabel("缓解增益 (%)")
    ax.set_title("幻觉缓解增益对比")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["RAG", "RAG+引用", "RAG+自检", "完整方案"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"缓解增益图已保存至: {save_path}")


def plot_type_heatmap(results: list[dict], save_path: str | Path = "data/results/type_heatmap.png"):
    """按题型幻觉率热力图"""
    from collections import defaultdict

    type_stats = defaultdict(lambda: defaultdict(lambda: {"hallucination": 0, "total": 0}))
    for r in results:
        qtype = r.get("question_type", "未知")
        model = r["model"]
        type_stats[qtype][model]["total"] += 1
        if "无法确定" not in r.get("response", ""):
            type_stats[qtype][model]["hallucination"] += 1

    types = sorted(type_stats.keys())
    models = sorted(set(r["model"] for r in results))

    data = np.zeros((len(types), len(models)))
    for i, qtype in enumerate(types):
        for j, model in enumerate(models):
            stats = type_stats[qtype][model]
            data[i][j] = stats["hallucination"] / stats["total"] * 100 if stats["total"] > 0 else 0

    fig, ax = plt.subplots(figsize=(len(models) * 3 + 4, len(types) * 1.5 + 2))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    for i in range(len(types)):
        for j in range(len(models)):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                    fontsize=10, color="white" if data[i, j] > 50 else "black")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types)
    ax.set_title("各题型幻觉率分布（% 越高越差）")

    fig.colorbar(im, ax=ax, shrink=0.8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"题型热力图已保存至: {save_path}")


def plot_all(metrics: dict, results: list[dict] | None = None, out_dir: str | Path = "data/results"):
    """生成全部图表"""
    out_dir = Path(out_dir)
    plot_hallucination_rate(metrics, out_dir / "hr_comparison.png")
    plot_mitigation_gain(metrics, out_dir / "mg_comparison.png")
    if results:
        plot_type_heatmap(results, out_dir / "type_heatmap.png")
    print("全部图表生成完毕。")
