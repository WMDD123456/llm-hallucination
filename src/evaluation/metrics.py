"""6 个评价指标计算 —— 使用 LLM-as-judge"""

import json
import time
from collections import defaultdict
from tqdm import tqdm

JUDGE_SYSTEM_PROMPT = "你是一个严谨的评分助手。请判断学生的回答是否与标准答案一致。"

JUDGE_TEMPLATE = """标准答案：{standard_answer}
学生回答：{student_response}

请判断学生回答的核心事实是否与标准答案一致。允许措辞差异和补充信息。
如果学生回答明确拒答（"无法确定""不知道""对不起"等），判为拒答。
仅回复一个 JSON：{{"correct": true/false, "is_refusal": true/false, "reason": "一句话说明"}}"""


def compute_metrics(results: list[dict], judge_api=None) -> dict:
    """
    计算所有指标，按 (model, group) 分组。

    参数:
        results: 实验结果列表
        judge_api: ModelAPI 实例，用于 LLM-as-judge 评测。为 None 时回退到关键词匹配。

    返回:
        { "deepseek-chat": { "G1": {"HR": ..., "FA": ..., ...}, "G2": {...} }, ... }
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["model"]][r["group"]].append(r)

    metrics = {}
    for model, groups in grouped.items():
        metrics[model] = {}
        for group, items in groups.items():
            metrics[model][group] = _compute_group_metrics(items, judge_api)
    return metrics


def _compute_group_metrics(items: list[dict], judge_api=None) -> dict:
    total = len(items)
    hallucination_count = 0
    fully_correct_count = 0
    refusal_count = 0

    for item in items:
        response = item.get("response", "")
        std_answer = item.get("standard_answer", "")

        if judge_api is not None:
            is_correct, is_refusal = _llm_judge(judge_api, response, std_answer)
        else:
            is_correct = _simple_judge(response, std_answer)
            is_refusal = "无法确定" in response

        if is_refusal:
            refusal_count += 1
        elif not is_correct:
            hallucination_count += 1
        else:
            fully_correct_count += 1

    hr = hallucination_count / total * 100
    fa = fully_correct_count / total * 100
    ra = refusal_count / total * 100

    return {
        "total": total,
        "hallucination_count": hallucination_count,
        "correct_count": fully_correct_count,
        "HR": round(hr, 2),
        "FA": round(fa, 2),
        "RA": round(ra, 2),
    }


def _llm_judge(api, student_response: str, standard_answer: str) -> tuple[bool, bool]:
    """用 LLM 判断回答是否正确。返回 (is_correct, is_refusal)。"""
    prompt = JUDGE_TEMPLATE.format(
        standard_answer=standard_answer,
        student_response=student_response,
    )
    try:
        raw = api.generate(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        # 提取 JSON（模型可能包裹在 ```json ... ``` 中）
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result.get("correct", False), result.get("is_refusal", False)
    except Exception:
        return _simple_judge(student_response, standard_answer), "无法确定" in student_response


def _simple_judge(response: str, standard_answer: str) -> bool:
    """回退方案：检查标准答案关键词是否出现在回答中"""
    keywords = standard_answer.strip().rstrip("。，；.!?？").split()
    match_count = sum(1 for kw in keywords if kw in response)
    return match_count >= len(keywords) * 0.5 if keywords else False


def compute_mitigation_gain(metrics: dict) -> dict:
    """计算缓解增益 MG = (G1_HR - Gx_HR) / G1_HR * 100%"""
    gains = {}
    for model, groups in metrics.items():
        baseline_hr = groups.get("G1", {}).get("HR", 0)
        gains[model] = {}
        for group, m in groups.items():
            if group == "G1":
                gains[model][group] = 0
            elif baseline_hr > 0:
                gains[model][group] = round((baseline_hr - m["HR"]) / baseline_hr * 100, 2)
            else:
                gains[model][group] = 0
    return gains


def print_report(metrics: dict, gains: dict):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("幻觉检测实验评估报告")
    print("=" * 70)

    for model in metrics:
        print(f"\n📊 模型: {model}")
        print("-" * 50)
        print(f"{'实验组':<30} {'HR(%)':<10} {'FA(%)':<10} {'MG(%)':<10}")
        print("-" * 50)
        for group in ["G1", "G2", "G3", "G4", "G5"]:
            if group in metrics[model]:
                m = metrics[model][group]
                g = gains.get(model, {}).get(group, 0)
                print(f"{group:<30} {m['HR']:<10} {m['FA']:<10} {g:<10}")
        print("-" * 50)

    print("\n" + "=" * 70)
    print("HR = 幻觉率 | FA = 事实准确率 | MG = 缓解增益")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    from src.experiments.runner import load_results

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        paths = sorted(Path("data/results").glob("results_*.json"))
        path = paths[-1] if paths else None
        if path is None:
            print("未找到结果文件")
            sys.exit(1)

    results = load_results(path)
    metrics = compute_metrics(results)
    gains = compute_mitigation_gain(metrics)
    print_report(metrics, gains)
