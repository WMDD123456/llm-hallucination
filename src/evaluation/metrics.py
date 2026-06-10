"""幻觉检测评测指标 —— LLM-as-judge + G3/G4 集成结果处理"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

JUDGE_SYSTEM_PROMPT = "你是一个公正的评分助手。判断学生回答的核心事实是否与标准答案一致。措辞差异、同义词、简写、补充信息均视为正确。"

JUDGE_TEMPLATE = """标准答案：{standard_answer}
学生回答：{student_response}

请判断学生回答的核心事实是否与标准答案一致。
判断规则：
- 核心事实相同即可，不要求措辞完全一致
- 同义词、简称、缩写、包含标准答案的超集均视为正确
- 例："豆沙"与"豆沙馅"→正确；"1920年"与"1920"→正确
- 例：详细解释中包含标准答案的关键事实→正确
- 只有完全偏离事实才判错误
- 明确拒答（"无法确定""暂无可靠资料"等）判为拒答

如果判定为错误（correct=false 且 is_refusal=false），还需对幻觉进行分类（5选1）：
- 事实错误：回答中的事实与标准答案不符（如人名、地名、作品名等实体错误）
- 关系错误：回答中实体之间的关系描述错误（如归属、因果、从属关系）
- 数值/时间错误：回答中的数字、日期、年份、数量等与标准答案不一致
- 信息编造：回答中的内容在标准答案和相关资料中完全找不到依据，属于凭空生成
- 冲突/逻辑矛盾：回答内部存在自相矛盾，或与常识/已知事实冲突

仅回复 JSON：{{"correct": true/false, "is_refusal": true/false, "hallucination_type": "事实错误/关系错误/数值或时间错误/信息编造/冲突或逻辑矛盾/无", "reason": "一句话"}}"""


def compute_metrics(results: list[dict], judge_api=None, cross_judges: list[str] = None) -> dict:
    """
    计算所有指标，按 (model, group) 分组。

    参数:
        results: 实验结果列表
        judge_api: 单个 ModelAPI 实例，用于 LLM-as-judge 评测。为 None 时回退到关键词匹配。
        cross_judges: 模型名列表，启用交叉评测：用A评B，用B评A，消除自评偏差。
    """
    from src.models.api import ModelAPI

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["model"]][r["group"]].append(r)

    # 裁判选择：只用 DeepSeek-Chat（铁律：GLM 幻觉率 53% 不能当裁判，
    #   Qwen 也没验证过裁判能力。DeepSeek 自评虽有放水偏差，
    #   但至少评测标准一致，横向可比）
    judge_map = {}
    PREFERRED_JUDGE = "deepseek-chat"
    if cross_judges and len(cross_judges) >= 2:
        if PREFERRED_JUDGE in cross_judges:
            judge_api_instance = ModelAPI(PREFERRED_JUDGE)
            for model in grouped:
                judge_map[model] = judge_api_instance
            print(f"[评测] 统一使用 {PREFERRED_JUDGE} 作为 LLM 裁判"
                  f"（{len(grouped)} 个被测模型）")

    metrics = {}
    for model, groups in grouped.items():
        metrics[model] = {}
        j = judge_map.get(model, judge_api)
        for group, items in groups.items():
            metrics[model][group] = _compute_group_metrics(items, j)
    return metrics


def _extract_final_from_ratcot(text: str) -> str | None:
    """从结构化推理文本中提取最终答案部分。

    各模型输出格式不一：
    - DeepSeek:  【最终答案】\\n答案内容
    - GLM-4:     【步骤四：形成最终答案】\\n答案内容
    - Qwen:      #### 最终答案\\n答案内容  或  最终答案：答案内容

    返回提取的答案文本，或 None 表示提取失败。
    """
    if not text:
        return None

    patterns = [
        r'【最终答案】\s*\n?\s*(.+)',
        r'【步骤\d+[：:].*?最终答案】\s*\n?\s*(.+)',
        r'【步骤\d+[：:].*?答案】\s*\n?\s*(.+)',
        r'\n\s*最终答案[：:]\s*(.+?)(?:\n\s*$|$)',
        r'####\s*最终答案\s*\n(.+)',
        r'###\s*最终答案\s*\n(.+)',
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            ans = matches[-1].group(1).strip()
            # 截断到下一个章节标记
            ans = re.split(r'\n\s*【|\n\s*#{1,4}\s|\n\s*---\s*\n', ans)[0]
            ans = ans.strip().rstrip('。，；.!?！？;；')
            if len(ans) > 3:
                return ans

    # 回退：找文本末尾包含"答案"或总结词的最后段落
    lines = text.strip().split('\n')
    for i in range(len(lines) - 1, max(len(lines) - 6, -1), -1):
        line = lines[i].strip()
        if ('答案' in line or '综上' in line) and len(line) > 10:
            rest = '\n'.join(lines[i:])
            rest = re.sub(r'^[^\n]*?[：:]\s*', '', rest)
            if len(rest) > 3:
                return rest[:300]

    return None


def _extract_response_for_judge(item: dict) -> str:
    """提取用于评测的回答文本。

    G4: 从 g4_final_answer 中提取【最终答案】部分，
         避免把整段推理过程送评
    G3: 直接使用 response
    G1/G2: 直接使用 response
    """
    group = item.get("group", "")
    response = item.get("response", "")

    if group == "G3":
        # G3 异构集成：优先用投票后的最终答案
        if item.get("g3_is_refusal", False):
            return "暂无可靠资料"
        final = item.get("g3_final_answer", "")
        if final and final != "暂无可靠资料":
            return final
        return response

    if group == "G4":
        # G4 拒答
        if item.get("g4_is_refusal", False):
            return "暂无可靠资料"

        # 优先从 g4_final_answer 提取最终答案
        final = item.get("g4_final_answer", "")
        extracted = _extract_final_from_ratcot(final)
        if extracted:
            return extracted

        # 其次从 response 提取
        extracted = _extract_final_from_ratcot(response)
        if extracted:
            return extracted

        # 回退：用 response 的最后一部分
        m = re.search(r'【最终答案】\s*(.+?)$', response, re.DOTALL)
        if m:
            return m.group(1).strip()

        return response

    # G3/G1/G2 直接返回
    return response


def _compute_group_metrics(items: list[dict], judge_api=None) -> dict:
    total = len(items)
    hallucination_count = 0
    fully_correct_count = 0
    refusal_count = 0
    hallucination_types = {
        "事实错误": 0,
        "关系错误": 0,
        "数值/时间错误": 0,
        "信息编造": 0,
        "冲突/逻辑矛盾": 0,
    }

    for item in items:
        response = _extract_response_for_judge(item)
        std_answer = item.get("standard_answer", "")
        group = item.get("group", "")

        # G4 优先用管线标记的拒答
        if group == "G4" and item.get("g4_is_refusal", False):
            refusal_count += 1
            continue

        is_correct = False
        is_refusal = False
        htype = ""

        if judge_api is not None:
            is_correct, is_refusal, htype = _llm_judge(judge_api, response, std_answer)
        else:
            is_correct = _simple_judge(response, std_answer)
            is_refusal = any(kw in response for kw in [
                "无法确定", "不知道", "暂无可靠资料", "无可靠资料", "审核未通过"
            ])

        if is_refusal:
            refusal_count += 1
        elif not is_correct:
            hallucination_count += 1
            if htype in hallucination_types:
                hallucination_types[htype] += 1
        else:
            fully_correct_count += 1

    hr = hallucination_count / total * 100 if total > 0 else 0
    fa = fully_correct_count / total * 100 if total > 0 else 0
    ra = refusal_count / total * 100 if total > 0 else 0
    answered = fully_correct_count + hallucination_count
    aa = fully_correct_count / answered * 100 if answered > 0 else 0

    return {
        "total": total,
        "hallucination_count": hallucination_count,
        "correct_count": fully_correct_count,
        "refusal_count": refusal_count,
        "HR": round(hr, 2),
        "FA": round(fa, 2),
        "RA": round(ra, 2),
        "AA": round(aa, 2),
        "hallucination_types": hallucination_types,
    }


def _llm_judge(api, student_response: str, standard_answer: str) -> tuple[bool, bool, str]:
    """用 LLM 判断回答是否正确。返回 (is_correct, is_refusal, hallucination_type)。"""
    prompt = JUDGE_TEMPLATE.format(
        standard_answer=standard_answer,
        student_response=student_response,
    )
    try:
        raw = api.generate(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return (
            result.get("correct", False),
            result.get("is_refusal", False),
            result.get("hallucination_type", "无"),
        )
    except Exception:
        is_correct = _simple_judge(student_response, standard_answer)
        is_refusal = any(
            kw in student_response for kw in ["无法确定", "不知道", "暂无可靠资料"]
        )
        return is_correct, is_refusal, ""


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
        print(f"\n[模型: {model}]")
        print("-" * 55)
        print(f"{'实验组':<25} {'HR(%)':<10} {'FA(%)':<10} {'RA(%)':<10} {'AA(%)':<10} {'MG(%)':<10}")
        print("-" * 55)
        for group in ["G1", "G2", "G3", "G4"]:
            if group in metrics[model]:
                m = metrics[model][group]
                g = gains.get(model, {}).get(group, 0)
                print(f"{group:<25} {m['HR']:<10} {m['FA']:<10} {m.get('RA', 0):<10} {m.get('AA', 0):<10} {g:<10}")
        print("-" * 55)

    # 幻觉类型分布
    print("\n" + "=" * 70)
    print("幻觉类型分布（5类）")
    print("=" * 70)
    TYPE_LABELS = ["事实错误", "关系错误", "数值/时间错误", "信息编造", "冲突/逻辑矛盾"]
    for model in metrics:
        for group in ["G1", "G2", "G3", "G4"]:
            if group in metrics[model]:
                m = metrics[model][group]
                htypes = m.get("hallucination_types", {})
                if htypes:
                    total_h = m.get("hallucination_count", 0)
                    print(f"\n[{model} / {group}] 幻觉总数: {total_h}")
                    for t in TYPE_LABELS:
                        cnt = htypes.get(t, 0)
                        pct = cnt / total_h * 100 if total_h > 0 else 0
                        bar = "█" * int(pct / 5)
                        print(f"  {t:<12} {cnt:>3} ({pct:5.1f}%) {bar}")
                    if total_h == 0:
                        print("  （无幻觉）")

    print("\n" + "=" * 70)
    print("HR = 幻觉率 | FA = 事实准确率 | RA = 拒答率 | AA = 敢答准确率 | MG = 缓解增益")
    print("=" * 70)


def compute_stratified_metrics(results: list[dict], judge_api=None, cross_judges: list[str] = None) -> dict:
    """按题型分层计算指标"""
    from src.models.api import ModelAPI

    judge_map = {}
    PREFERRED_JUDGE = "deepseek-chat"
    if cross_judges and len(cross_judges) >= 2 and PREFERRED_JUDGE in cross_judges:
        judge_api_instance = ModelAPI(PREFERRED_JUDGE)
        for model in set(r["model"] for r in results):
            judge_map[model] = judge_api_instance

    strata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        qtype = r.get("question_type", "未分类")
        model = r["model"]
        group = r["group"]
        strata[model][group][qtype].append(r)

    stratified = {}
    for model, groups in strata.items():
        stratified[model] = {}
        j = judge_map.get(model, judge_api)
        for group, types in groups.items():
            stratified[model][group] = {}
            for qtype, items in types.items():
                stratified[model][group][qtype] = _compute_group_metrics(items, j)

    return stratified


def print_stratified_report(stratified: dict, group: str = "G4"):
    """打印分层评测报告"""
    print("\n" + "=" * 70)
    print(f"难度分层评测报告（{group}）")
    print("=" * 70)

    for model in stratified:
        if group not in stratified[model]:
            continue
        print(f"\n[模型: {model}]")
        print("-" * 50)
        print(f"{'题型':<20} {'题目数':<10} {'HR(%)':<10} {'FA(%)':<10}")
        print("-" * 50)
        for qtype, m in sorted(stratified[model][group].items()):
            print(f"{qtype:<20} {m['total']:<10} {m['HR']:<10} {m['FA']:<10}")
        print("-" * 50)


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
