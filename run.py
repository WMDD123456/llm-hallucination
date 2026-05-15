#!/usr/bin/env python
"""主入口：一键运行完整实验流程

用法:
    python run.py                    # 运行完整实验
    python run.py --models deepseek  # 指定模型
    python run.py --groups G1 G2     # 指定实验组
    python run.py --eval-only        # 仅评估已有结果
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    QUESTIONS_FILE, KB_DIR, KB_INDEX_DIR, RAG_CONFIG, get_available_models
)
from src.experiments.runner import run_experiment, save_results
from src.evaluation.metrics import compute_metrics, compute_mitigation_gain, print_report
from src.visualization.plots import plot_all


def step1_build_kb():
    """Step 1: 构建知识库索引"""
    try:
        from src.rag.build_kb import build_knowledge_base
    except ImportError as e:
        print(f"[Step 1] 跳过知识库构建（缺少依赖: {e}）")
        return False

    data_file = KB_DIR / "wiki_data.json"
    if not data_file.exists():
        print(f"知识库数据文件不存在: {data_file}")
        return False

    print("[Step 1] 构建知识库索引...")
    try:
        build_knowledge_base(
            data_file=data_file,
            index_dir=KB_INDEX_DIR,
            embedding_model=RAG_CONFIG["embedding_model"],
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"],
        )
        return True
    except Exception as e:
        print(f"[Step 1] 知识库构建失败: {e}")
        print("[Step 1] 可跳过此步骤，仅运行 G1 基线: py run.py --skip-kb --groups G1")
        return False


def step2_run_experiments(models, groups):
    """Step 2: 运行实验"""
    import json

    if not QUESTIONS_FILE.exists():
        print(f"题目文件不存在: {QUESTIONS_FILE}")
        return None

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[Step 2] 加载 {len(questions)} 道题目")

    retriever = None
    if any(g != "G1" for g in groups):
        try:
            from src.rag.retriever import get_retriever
            retriever = get_retriever(KB_INDEX_DIR)
        except Exception as e:
            print(f"[Step 2] 检索器初始化失败: {e}")
            print("[Step 2] 将跳过 RAG 相关实验组，仅运行 G1")
            groups = [g for g in groups if g == "G1"]
            if not groups:
                print("[Step 2] 没有可运行的实验组")
                return None

    results = run_experiment(
        questions=questions,
        models=models,
        groups=groups,
        retriever=retriever,
    )

    path = save_results(results, label="_".join(models[:2]))
    return path


def step3_evaluate(results_path):
    """Step 3: 评估与可视化（使用 LLM-as-judge）"""
    from src.experiments.runner import load_results
    from src.models.api import ModelAPI

    print(f"[Step 3] 加载实验结果...")
    results = load_results(results_path)

    # 用第一个可用模型作为裁判
    available = get_available_models()
    if available:
        judge = ModelAPI(available[0])
        print(f"[Step 3] 使用 {available[0]} 作为 LLM 裁判...")
    else:
        judge = None
        print("[Step 3] 无可用的裁判模型，回退到关键词匹配")

    metrics = compute_metrics(results, judge_api=judge)
    gains = compute_mitigation_gain(metrics)
    print_report(metrics, gains)
    plot_all(metrics, results)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="大模型幻觉检测实验")
    parser.add_argument("--models", nargs="+", default=None, help="指定模型，如 deepseek-chat qwen-plus")
    parser.add_argument("--groups", nargs="+", default=None, help="指定实验组，如 G1 G2")
    parser.add_argument("--eval-only", type=str, default=None, help="仅评估指定的结果文件")
    parser.add_argument("--skip-kb", action="store_true", help="跳过知识库构建")
    args = parser.parse_args()

    # 仅评估模式
    if args.eval_only:
        step3_evaluate(args.eval_only)
        return

    # 检查可用模型
    available = get_available_models()
    if not available:
        print("\n❌ 没有可用的模型！")
        print("请在 .env 文件中配置至少一个 API key，然后运行:")
        print("  cp .env.example .env")
        print("  # 编辑 .env 文件填入你的 API key")
        return

    print(f"可用模型: {available}")

    models = args.models or available
    groups = args.groups or ["G1", "G2"]
    models = [m for m in models if m in available]
    if not models:
        print("指定的模型均不可用")
        return

    print(f"实验计划: {len(models)} 个模型 × {len(groups)} 个实验组")

    # Step 1: 构建知识库
    if not args.skip_kb and any(g != "G1" for g in groups):
        success = step1_build_kb()
        if not success:
            print("回退为仅运行 G1 基线（不使用 RAG）\n")
            groups = ["G1"]

    # Step 2: 运行实验
    results_path = step2_run_experiments(models, groups)
    if results_path is None:
        return

    # Step 3: 评估
    step3_evaluate(results_path)

    print("\n✅ 实验完成！")
    print(f"结果文件: {results_path}")


if __name__ == "__main__":
    main()
