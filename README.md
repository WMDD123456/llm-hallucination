# 基于事实性问答场景的大模型幻觉检测、评测与缓解方法研究

## 一、研究背景

大语言模型存在"幻觉"问题——生成的内容语法通顺但与事实不符。在医疗、法律、教育等对事实准确性要求高的场景中，幻觉严重限制了 LLM 的实际应用。

本课题针对中文事实性问答场景，设计四组消融实验（G1-G4），逐步叠加缓解策略，系统评估不同方案的效果。

## 二、方案设计

### 整体架构

```
评测集 (150题) ──→ G1 裸模 ──→ G2 +基础RAG ──→ G3 +异构集成 ──→ G4 全链路
                     │              │                 │                │
                     ▼              ▼                 ▼                ▼
                LLM-as-Judge 统一评测 (DeepSeek-Chat 裁判)
                     │
                     ▼
              HR / FA / RA 三指标
```

### 四组消融实验

| 实验组 | 策略 | 模型 | 说明 |
|--------|------|------|------|
| G1 | 裸模型基线 | DeepSeek-Chat ×1 | 直接提问，无 RAG，无干预 |
| G2 | +基础 RAG | DeepSeek-Chat ×1 | 检索百度百科 evidence 注入 prompt |
| G3 | +异构集成 | DS + GLM + Qwen-Max + Qwen-Turbo | 4 模型各自查资料回答，相似度加权投票 |
| G4 | 全链路 | 同上 4 模型 | 路由分类 → CoT 推理 → JSON 生成 → 投票 → 不确定性拒答 |

**对比逻辑**：
- G1 vs G2：RAG 单独有效吗？
- G2 vs G3：多模型投票比单模型 RAG 更好吗？
- G3 vs G4：加路由 + CoT + 拒答有额外收益吗？

## 三、评测方法

### 评测指标

| 指标 | 含义 | 方向 |
|------|------|------|
| HR (Hallucination Rate) | 幻觉回答占比 | ↓ 越低越好 |
| FA (Factual Accuracy) | 事实正确占比 | ↑ 越高越好 |
| RA (Refusal Rate) | 拒答占比 | — 适当即可 |

### LLM-as-Judge

- 裁判模型：**DeepSeek-Chat**（统一裁判，避免不一致）
- Prompt：宽松版，容忍同义词、简写、合理超集
- 判定：每道题判为"正确 / 幻觉 / 拒答"

## 四、实验结果（2026-06-08）

### 全量 150 题

| 组别 | HR ↓ | FA ↑ | RA | 敢答准确率 |
|------|------|------|-----|-----------|
| **G1** 裸模 | 20.0% | 77.3% | 2.7% | 79% |
| **G2** +RAG | **0.7%** | **83.3%** | 16.0% | **99%** |
| **G3** +异构集成 | 1.3% | 81.3% | 17.3% | 98% |
| **G4** 全链路 | 4.0% | 79.3% | 16.7% | 95% |

### 复杂推理题子集 (10 题)

| 组别 | FA ↑ | RA |
|------|------|-----|
| G2 | 50.0% | 50.0% |
| **G4** | **70.0%** | 30.0% |

### 关键发现

1. **RAG 是最大贡献因子**：G1→G2 幻觉率从 20% 降至 0.7%（降幅 96.5%），"不知道就拒答"远好于"瞎编"
2. **多模型投票不是万能药**：数据集 93% 为简单事实题，G3/G4 的投票反而内耗；但在 10 道复杂推理题上 G4 显著优于 G2（FA +20pp）
3. **策略应与题型匹配**：简单题用 G2（RAG 足够），复杂推理题用 G4（多视角验证）

## 五、技术栈

| 层次 | 选型 |
|------|------|
| LLM 调用 | DeepSeek-Chat, GLM-4-Flash, Qwen-Max, Qwen-Turbo |
| RAG 检索 | FAISS + bge-large-zh-v1.5 + 百度百科 300 篇词条 |
| 自动评测 | LLM-as-Judge (DeepSeek-Chat) |
| 集成投票 | Jaccard 相似度加权投票 + 不确定性阈值拒答 |
| 可视化 | matplotlib |

## 六、项目结构

```
llm-hallucination/
├── run.py                      # 主入口，一键运行实验
├── eval_llm_judge.py           # LLM-as-Judge 评测
├── visualize_results.py        # 可视化脚本
├── src/
│   ├── config.py               # 全局配置（模型、路径、参数）
│   ├── models/api.py           # 统一 LLM API 调用
│   ├── experiments/
│   │   ├── runner.py           # G1-G4 实验运行器
│   │   └── prompts.py          # 各组 Prompt 模板
│   ├── rag/
│   │   ├── build_kb.py         # 知识库构建
│   │   └── retriever.py        # FAISS 检索器
│   ├── evaluation/
│   │   └── metrics.py          # 指标计算 + LLM-Judge
│   └── visualization/
│       └── plots.py            # matplotlib 图表
├── data/
│   ├── eval_set_150.json       # 150 题评测集
│   ├── knowledge_base/         # 百度百科语料
│   └── results/                # 实验结果 JSON
├── .env.example                # API key 配置模板
└── requirements.txt
```

## 七、运行

```bash
pip install -r requirements.txt
cp .env.example .env   # 编辑填入 API key

# 全部实验
python run.py --groups G1 G2 G3 G4

# 仅评测已有结果
python eval_llm_judge.py data/results/results_xxx.json

# 画图
python visualize_results.py
```

## 八、分工

| 成员 | 职责 |
|------|------|
| 王明达 | 模型调用、RAG 系统、实验执行、数据分析与可视化 |
| 邓钧尹 | 评测集构建、标注与审核 |
| 黄春兵 | PPT 制作、汇报 |
