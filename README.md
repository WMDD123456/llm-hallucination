# 基于事实性问答场景的大模型幻觉检测、评测与缓解方法研究

## 一、研究背景与意义

大语言模型（LLM）在生成文本时可能产生"幻觉"（Hallucination）——生成的内容语法通顺但与事实不符。在医疗、法律、教育等对事实准确性要求高的场景中，幻觉问题严重限制了 LLM 的实际应用。

现有幻觉缓解方案主要包括：检索增强生成（RAG）、思维链推理、自我校验等。但这些方案在中文事实性问答场景中的效果如何、哪种策略组合最优，仍缺乏系统的消融实验对比。

本课题针对上述问题，构建中文事实性问答评测集，设计五组消融实验，系统评估不同缓解策略在 DeepSeek、GLM 等主流中文 LLM 上的效果。

## 二、研究目标

1. 构建高质量中文事实性问答幻觉评测集（150 题，覆盖多领域多题型）
2. 实现 G1-G5 五组消融实验框架，对比不同缓解策略的效果
3. 使用 LLM-as-judge 方法替代传统关键词匹配，提升评测准确性
4. 通过多模型对比，分析不同 LLM 的幻觉倾向差异

## 三、技术路线

```
数据集构建                   实验执行                      评测分析
┌──────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Chinese-SQA  │      │  G1  直接问答     │      │  LLM-as-judge   │
│  (120题)     │      │  G2  +RAG检索     │      │  (自动评测)      │
│              │─────►│  G3  +证据引用    │─────►│                 │
│  WebQA       │      │  G4  +Self-check  │      │  HR/FA/RA/MG    │
│  (30题)      │      │  G5  全套方案     │      │  指标计算        │
└──────────────┘      └──────────────────┘      └─────────────────┘
       ▲                        ▲                        │
       │                        │                        ▼
  标注规则               RAG 知识库              ┌─────────────────┐
  人工审核              (FAISS索引)              │  matplotlib     │
                                                  │  可视化图表      │
                                                  └─────────────────┘
```

**核心组件**:

| 层次 | 技术选型 | 说明 |
|------|----------|------|
| LLM 调用 | DeepSeek-Chat, GLM-4-Flash | 统一通过 OpenAI 兼容接口 |
| 向量检索 | LangChain + FAISS + bge-large-zh-v1.5 | 中文 Embedding，CPU 可运行 |
| 自动评测 | LLM-as-judge + 关键词匹配回退 | 用 LLM 判断回答事实准确性 |
| 可视化 | matplotlib | 幻觉率对比柱状图、题型热力图 |

## 四、实验设计

### 4.1 五组消融实验

| 实验组 | 策略 | Prompt 设计 |
|--------|------|------------|
| G1 | 原生生成 (Baseline) | 直接提问，无参考材料 |
| G2 | RAG 检索增强 | 检索 Top-3 相关段落注入 prompt |
| G3 | RAG + 强制证据引用 | G2 基础上要求用 `[来源N]` 标注引用 |
| G4 | RAG + Self-check | G2 基础上要求先回答再自我核实 |
| G5 | 完整方案 | RAG + 证据引用 + Self-check 组合 |

**实验假设**: RAG 能降低幻觉率（G2 vs G1）；证据引用和 Self-check 各自有额外收益（G3/G4 vs G2）；三者组合效果最优（G5）。

### 4.2 评测指标

| 指标 | 公式 / 含义 |
|------|------------|
| HR (Hallucination Rate) | 幻觉回答数 / 总题数 × 100% |
| FA (Factual Accuracy) | 完全正确回答数 / 总题数 × 100% |
| RA (Refusal Rate) | 拒答数 / 总题数 × 100% |
| MG (Mitigation Gain) | (G1_HR - Gx_HR) / G1_HR × 100%，衡量缓解效果 |

**判定标准**:
- **正确**: 核心事实与标准答案一致，允许措辞差异
- **幻觉**: 核心事实与标准答案不一致（错误、无中生有、张冠李戴）
- **拒答**: 模型明确表示无法回答

## 五、评测集构建

从两个公开数据集分层抽样构建 150 题评测集：

| 来源 | 题目数 | 覆盖范围 |
|------|--------|----------|
| [Chinese-SimpleQA](https://github.com/OpenStellarTeam/ChineseSimpleQA) | 120 | 6 大领域（科技、历史、地理、文化、生物、医药等）99 个子类别 |
| [WebQA](https://github.com/WebQnA/WebQA) | 30 | 百度知道日常问答，附带采纳回答 evidence |

题型分布由 `classify_type()` 自动判定：事实判断题、数值/时间题、多实体关系题。

每道题字段：`id, question, answer, type, category, source, evidence, label_status, reviewer_notes`

**构建脚本**: `build_eval_set.py` | **标注规则**: `data/eval_set_150_annotation_rules.json` | **状态**: 已审核

## 六、当前实验结果

**评测集**: eval_set_150 (150 题) | **实验组**: G1 基线 | **评测方式**: 关键词匹配

| 模型 | HR (幻觉率 ↓) | FA (准确率 ↑) | RA (拒答率) |
|------|-------------|-------------|-----------|
| DeepSeek-Chat | 26.00% | 70.67% | 3.33% |
| GLM-4-Flash  | 58.00% | 37.33% | 4.67% |

**分析**:
- DeepSeek-Chat 在中文事实性问答上显著优于 GLM-4-Flash（幻觉率低 32 个百分点）
- 两个模型均存在幻觉，评测集能有效检测幻觉问题
- 基线已确立，G2-G5 缓解实验有明确的对比参照
- 当前评测回退到关键词匹配（LLM-as-judge 代码已就绪，正式评测待执行）

## 七、项目结构

```
llm-hallucination/
├── run.py                  # 主入口，一键运行实验流程
├── build_eval_set.py       # 评测集构建脚本
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明（本文件）
├── data/
│   ├── eval_set_150.json               # 150 题幻觉评测集
│   ├── eval_set_150_annotation_rules.json  # 标注规则
│   ├── questions.json                  # Demo 题目（20 题）
│   ├── questions_chinese_simpleqa.json # Chinese-SimpleQA 题目
│   ├── questions_webqa.json            # WebQA 题目
│   ├── knowledge_base/
│   │   └── wiki_data.json             # 知识库语料（RAG 检索源）
│   └── results/
│       └── results_*.json             # 实验结果
├── src/
│   ├── config.py           # 全局配置（模型、路径、RAG 参数）
│   ├── models/
│   │   └── api.py          # 统一 LLM API 调用封装
│   ├── experiments/
│   │   ├── runner.py       # 实验运行器（遍历 模型×实验组×题目）
│   │   └── prompts.py      # G1-G5 Prompt 模板
│   ├── rag/
│   │   ├── build_kb.py     # 知识库向量索引构建
│   │   └── retriever.py    # FAISS 检索器
│   ├── evaluation/
│   │   └── metrics.py      # 指标计算 + LLM-as-judge 评测
│   └── visualization/
│       └── plots.py        # matplotlib 可视化
```

## 八、运行说明

### 环境要求

- Python 3.10+（本项目使用 Anaconda Python: `/c/ProgramData/anaconda3/python`）
- Windows / Linux / macOS，无需 GPU

### 安装

```bash
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入至少一个 API key:
#   DEEPSEEK_API_KEY=sk-xxx
#   ZHIPU_API_KEY=xxx
```

### 运行实验

```bash
# G1 基线（不需要 RAG 知识库）
python run.py --dataset eval_150 --skip-kb --groups G1

# 全部实验组（需先构建知识库）
python run.py --dataset eval_150 --groups G1 G2 G3 G4 G5

# 切换数据集
python run.py --dataset demo          # 20 题快速测试
python run.py --dataset chinese_simpleqa
python run.py --dataset webqa

# 指定模型
python run.py --models deepseek-chat glm-4-flash

# 仅评估已有结果
python run.py --eval-only data/results/results_xxx.json
```

数据集选项：`demo` / `chinese_simpleqa` / `webqa` / `eval_150`

## 九、已完成工作

| 模块 | 状态 | 说明 |
|------|------|------|
| 评测集构建 | ✅ 完成 | 150 题 + 标注规则，已审核 |
| LLM API 封装 | ✅ 完成 | DeepSeek + GLM 双模型统一接口 |
| G1 基线实验 | ✅ 完成 | 双模型 150 题跑通，HR 数据已采集 |
| 指标计算 | ✅ 完成 | HR/FA/RA/MG 四指标，LLM-as-judge 已实现 |
| RAG 检索链路 | ⚠️ 代码就绪 | FAISS + bge-large-zh，依赖已安装，索引待构建 |
| 可视化 | ✅ 完成 | 幻觉率对比柱状图 + 题型热力图 |
| Prompt 模板 | ✅ 完成 | G1-G5 五组 Prompt 已设计 |
| G2-G5 实验 | ⬜ 待执行 | 检索链路就绪，可随时运行 |
| LLM-as-judge 正式评测 | ⬜ 待执行 | 代码已就绪，需运行 |

## 十、后续计划

1. **G2-G5 消融实验**: 在 eval_150 上跑完全部五组，验证各策略的缓解效果
2. **评测集审核**: 已完成，150 题全部审核通过
3. **LLM-as-judge 正式评测**: 用 LLM 裁判替代关键词匹配，提高评测准确性
4. **多模型对比**: 完成双模型 × 五实验组的完整对比
5. **补充实验**: 探索更多缓解方案（如 CoVe 校验链、检索后重排序等）
6. **分析与报告**: 撰写实验分析报告，整理可视化结果，准备最终汇报

## 分工

| 成员 | 职责 |
|------|------|
| 王明达 | 模型调用、RAG 系统搭建、实验执行、数据采集与可视化 |
| 邓钧尹 | 评测集构建、标注、质量审核 |
| 黄春兵 | PPT 制作、最终汇报 |
