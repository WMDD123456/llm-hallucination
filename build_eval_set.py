#!/usr/bin/env python
"""从 Chinese-SimpleQA 和 WebQA 构建 150 题中文事实性问答幻觉评测集"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"


def classify_type(question: str, answer: str) -> str:
    """根据问句和答案自动判定题型"""
    q = question.strip()

    # 数值/时间题：问数量、年份、长度、重量、速度等
    num_patterns = [
        r'(多少|几[个只条座]?|哪[一]?年|何时|什么时候|多大|多高|多长|多重|多久|多快)',
        r'(\d+年|公元|世纪)',
        r'(数量|数值|长度|高度|重量|速度|面积|人口)',
    ]
    for pat in num_patterns:
        if re.search(pat, q):
            return "数值/时间题"

    # 多实体关系题：涉及多个实体之间的关联
    multi_patterns = [
        r'(和|与|及).*(关系|区别|联系|关联)',
        r'(属于|归类|分类).*(哪|什么)',
        r'(哪个.*位于|什么.*属于|谁.*属于)',
        r'(由|被).*(发现|发明|创建|提出|命名)',
        r'.*组成.*',
        r'(哪些|几种|几个).*',
    ]
    for pat in multi_patterns:
        if re.search(pat, q):
            return "多实体关系题"

    # 定义题：问概念、术语的定义、含义
    define_patterns = [
        r'(什么是|什么叫|是什么|指的是|是指|指的是什么)',
        r'(定义|含义|概念|全称|简称)',
    ]
    for pat in define_patterns:
        if re.search(pat, q):
            return "事实判断题"

    # 其余归为事实判断题
    return "事实判断题"


def build_from_chinese_simpleqa(n_total: int = 120) -> list[dict]:
    with open(DATA_DIR / "Chinese-SimpleQA.jsonl", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]

    # 按一级分类分组
    by_category = defaultdict(list)
    for q in raw:
        by_category[q["primary_category"]].append(q)

    # 每个类别按比例分配，至少 5 题
    cats = sorted(by_category.keys())
    print(f"Chinese-SimpleQA 类别数: {len(cats)}")
    for cat in cats:
        print(f"  {cat}: {len(by_category[cat])} 题")

    # 6 个类别，每个 ~20 题
    per_cat = n_total // len(cats)
    remainder = n_total % len(cats)

    selected = []
    for i, cat in enumerate(cats):
        pool = by_category[cat]
        n = per_cat + (1 if i < remainder else 0)
        # 在类别内按二级分类分层抽样
        by_subcat = defaultdict(list)
        for q in pool:
            by_subcat[q.get("secondary_category", "其他")].append(q)

        subcat_samples = []
        for subcat, items in by_subcat.items():
            # 每个子类取适量
            take = max(1, min(len(items), max(1, n * len(items) // len(pool))))
            subcat_samples.extend(random.sample(items, min(take, len(items))))

        # 如果不足 n，从整个类别随机补
        if len(subcat_samples) < n:
            remaining = [q for q in pool if q not in subcat_samples]
            subcat_samples.extend(random.sample(remaining, min(n - len(subcat_samples), len(remaining))))
        else:
            subcat_samples = random.sample(subcat_samples, n)

        selected.extend(subcat_samples)

    # 转换格式
    result = []
    for i, q in enumerate(selected):
        qtype = classify_type(q["question"], q["answer"])
        evidence = q.get("urls", [""])[0] if q.get("urls") else ""
        result.append({
            "id": i + 1,
            "question": q["question"],
            "answer": q["answer"],
            "type": qtype,
            "category": q["primary_category"],
            "subcategory": q.get("secondary_category", ""),
            "source": "Chinese-SimpleQA",
            "evidence": evidence,
            "label_status": "待人工审核",
            "reviewer_notes": "",
        })
    return result


def build_from_webqa(n_total: int = 30) -> list[dict]:
    with open(DATA_DIR / "WebQA/KD_WebQA_test.jsonl", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]

    # 只取有 evidence 的、题目长度合适的
    pool = []
    for q in raw:
        q_text = q["question"].strip()
        ev = q.get("evidence", "").strip()
        ans = q["answer"]
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        if not q_text or not ev or not ans:
            continue
        if len(q_text) < 5 or len(q_text) > 200:
            continue
        pool.append(q)

    print(f"WebQA 有效题数: {len(pool)}")

    # 随机采样
    selected = random.sample(pool, min(n_total, len(pool)))

    result = []
    for i, q in enumerate(selected):
        ans = q["answer"]
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        qtype = classify_type(q["question"], ans)
        result.append({
            "id": i + 1,
            "question": q["question"],
            "answer": ans,
            "type": qtype,
            "category": "百度知道",
            "subcategory": "",
            "source": "WebQA",
            "evidence": q.get("evidence", ""),
            "label_status": "待人工审核",
            "reviewer_notes": "",
        })
    return result


def main():
    print("=" * 60)
    print("构建 150 题中文事实性问答幻觉评测集")
    print("=" * 60)

    print("\n[1/3] 从 Chinese-SimpleQA 精选 120 题...")
    csqa_questions = build_from_chinese_simpleqa(120)

    print(f"\n[2/3] 从 WebQA 精选 30 题...")
    wqa_questions = build_from_webqa(30)

    all_questions = csqa_questions + wqa_questions

    # 重新编号
    for i, q in enumerate(all_questions):
        q["id"] = i + 1

    # 统计题型分布
    type_counts = defaultdict(int)
    cat_counts = defaultdict(int)
    for q in all_questions:
        type_counts[q["type"]] += 1
        cat_counts[q["category"]] += 1

    print(f"\n[3/3] 题型与类别分布:")
    print(f"  总题数: {len(all_questions)}")
    print(f"  题型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")
    print(f"  来源分布:")
    for s, c in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    # 保存
    output_path = DATA_DIR / "eval_set_150.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\n评测集已保存至: {output_path}")

    # 保存标注规则
    annotation_rules = {
        "评测集名称": "中文事实性问答幻觉评测集 (v1.0)",
        "题目数量": 150,
        "题目来源": {
            "Chinese-SimpleQA": {
                "数量": 120,
                "描述": "由 OpenStellarTeam 构建的高质量中文事实性问答数据集，题目涵盖 6 大领域 99 个子类别",
                "证据来源": "Wikipedia / 百度百科 URL",
                "网址": "https://github.com/OpenStellarTeam/ChineseSimpleQA",
            },
            "WebQA": {
                "数量": 30,
                "描述": "百度知道社区问答数据集，题目贴近日常，附带社区采纳的 evidence 文本",
                "证据来源": "百度知道采纳回答",
                "网址": "https://github.com/WebQnA/WebQA",
            },
        },
        "题型定义": {
            "事实判断题": "问句要求判定一个事实或给出实体名称，答案通常为名词短语（人名、地名、物名等）",
            "数值/时间题": "问句要求给出具体的数值、年份、日期、数量等",
            "多实体关系题": "问句涉及两个及以上实体之间的关系，或要求列举多项",
        },
        "答案判定标准": {
            "正确": "核心事实与标准答案一致，允许措辞差异和合理补充信息",
            "幻觉": "核心事实与标准答案不一致（错误、无中生有、张冠李戴）",
            "拒答": "模型明确表示无法回答（如'无法确定''不知道''对不起'）",
        },
        "人工审核要点": [
            "逐个核对题目与标准答案的正确性",
            "验证 evidence URL 是否可访问",
            "确认题目无歧义，参考答案无争议",
            "检查题型分类是否准确",
            "审核通过后将 label_status 改为'已审核'",
        ],
        "构建日期": "2026-05-16",
        "构建工具": "build_eval_set.py",
    }

    rules_path = DATA_DIR / "eval_set_150_annotation_rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(annotation_rules, f, ensure_ascii=False, indent=2)
    print(f"标注规则已保存至: {rules_path}")


if __name__ == "__main__":
    main()
