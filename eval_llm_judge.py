"""LLM-as-judge 评估脚本（带进度条）"""
import json, sys, time, shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation.metrics import compute_metrics, compute_mitigation_gain, print_report, _llm_judge
from src.experiments.runner import load_results
from src.models.api import ModelAPI
from src.config import get_available_models

# --- 加载 ---
results_path = sys.argv[1] if len(sys.argv) > 1 else sorted(
    Path("data/results").glob("results_*.json"))[-1]
print(f"结果文件: {results_path}")
results = load_results(results_path)
print(f"共 {len(results)} 条结果")

# --- 裁判模型 ---
available = get_available_models()
print(f"可用模型: {available}")
judge = ModelAPI(available[0])
print(f"裁判模型: {available[0]}")

# --- 逐条评估(带进度条) ---
grouped = defaultdict(lambda: defaultdict(list))
for r in results:
    grouped[r["model"]][r["group"]].append(r)

metrics = {}
for model_name, groups in grouped.items():
    metrics[model_name] = {}
    for group_name, items in groups.items():
        print(f"\n评估 {model_name} / {group_name} ({len(items)} 条)")
        total = len(items)
        hc = fc = rc = 0  # hallucination, fully correct, refusal counts

        for item in tqdm(items, desc=f"  {group_name}"):
            response = item.get("response", "")
            std = item.get("standard_answer", "")
            ok, refused, htype = _llm_judge(judge, response, std)
            if refused:
                rc += 1
            elif not ok:
                hc += 1
            else:
                fc += 1
            time.sleep(0.1)  # API 限流

        metrics[model_name][group_name] = {
            "total": total, "hallucination_count": hc,
            "correct_count": fc, "refusal_count": rc,
            "HR": round(hc / total * 100, 2),
            "FA": round(fc / total * 100, 2),
            "RA": round(rc / total * 100, 2),
        }

# --- 输出 ---
gains = compute_mitigation_gain(metrics)
print_report(metrics, gains)

metrics_path = Path("data/results/metrics_llm_judge.json")

# 保存前先备份旧文件
if metrics_path.exists():
    backup_dir = Path("data/results/backups")
    backup_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"metrics_llm_judge_backup_{ts}.json"
    shutil.copy(metrics_path, backup_path)
    print(f"\n旧指标已备份至 {backup_path}")

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({"metrics": metrics, "gains": gains}, f, ensure_ascii=False, indent=2)
print("指标已保存至 data/results/metrics_llm_judge.json")

from src.visualization.plots import plot_all
plot_all(metrics, results)
print("全部完成！")
