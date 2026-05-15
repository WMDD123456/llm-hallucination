"""实验运行器：遍历 题目 × 模型 × 实验组，采集结果"""

import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from src.config import RESULTS_DIR, get_available_models
from src.models.api import ModelAPI
from src.rag.retriever import get_retriever
from src.experiments.prompts import PROMPT_BUILDERS, GROUP_LABELS


def run_experiment(
    questions: list[dict],
    models: list[str] | None = None,
    groups: list[str] | None = None,
    retriever=None,
) -> list[dict]:
    """
    运行实验并返回所有结果。

    参数:
        questions: 题目列表，每题为 {"id", "question", "answer", "type", "evidence"}
        models: 要测试的模型列表，None 则使用所有可用模型
        groups: 要测试的实验组，None 则测试 G1-G5 全部
        retriever: Retriever 实例（G2-G5 需要）

    返回:
        结果列表，每条包含 question_id, model, group, prompt, response, ...
    """
    if models is None:
        models = get_available_models()
        if not models:
            raise RuntimeError("没有可用的模型。请在 .env 中配置至少一个 API key。")

    if groups is None:
        groups = list(PROMPT_BUILDERS.keys())

    results = []
    total = len(questions) * len(models) * len(groups)

    with tqdm(total=total, desc="实验进度") as pbar:
        for model_name in models:
            api = ModelAPI(model_name)

            for group in groups:
                build_prompt = PROMPT_BUILDERS[group]

                for q in questions:
                    if group == "G1":
                        prompt = build_prompt(q["question"])
                    else:
                        retrieved = retriever.retrieve(q["question"])
                        prompt = build_prompt(q["question"], retrieved)

                    try:
                        response = api.generate(prompt)
                    except Exception as e:
                        response = f"[ERROR] {e}"

                    results.append({
                        "question_id": q["id"],
                        "question": q["question"],
                        "standard_answer": q["answer"],
                        "question_type": q.get("type", ""),
                        "evidence": q.get("evidence", ""),
                        "model": model_name,
                        "group": group,
                        "prompt": prompt,
                        "response": response,
                    })

                    pbar.update(1)
                    time.sleep(0.3)  # 避免 API 限流

    return results


def save_results(results: list[dict], label: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"results_{label}_{timestamp}.json" if label else f"results_{timestamp}.json"
    path = RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {path}")
    return path


def load_results(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
