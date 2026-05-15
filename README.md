# 大语言模型幻觉检测与缓解方案研究

## 一、课题背景

大语言模型（LLM）在生成文本时存在"幻觉"（Hallucination）问题——模型会输出看似合理但事实错误的内容。这一问题严重制约了 LLM 在知识密集型场景中的可靠应用。

本课题设计了一套完整的幻觉检测与缓解实验框架，通过消融实验对比多种缓解策略的效果，量化不同方案对幻觉率的影响。

## 二、研究目标

1. 构建可控的幻觉检测实验环境，在多模型上复现幻觉现象
2. 实现并对比 5 种缓解策略：原生生成、RAG 检索增强、证据引用约束、Self-check 自检、完整组合方案
3. 建立科学的评估体系，使用 LLM-as-judge 代替关键词匹配进行幻觉判定
4. 通过消融实验量化各策略的缓解增益，为幻觉缓解方案选型提供数据支撑

## 三、技术方案

### 3.1 实验设计

设置 5 个实验组（消融实验），每组使用相同的 20 道事实型题目进行测试：

| 实验组 | 策略 | 说明 |
|--------|------|------|
| G1 | Baseline（原生生成） | 直接问答，无任何外部知识辅助 |
| G2 | RAG 检索增强 | 从维基百科知识库检索相关段落注入 prompt |
| G3 | RAG + 强制证据引用 | G2 基础上要求标注引用来源 |
| G4 | RAG + Self-check | G2 基础上要求模型自查事实准确性 |
| G5 | 完整方案 | RAG + 证据引用 + Self-check 叠加 |

### 3.2 技术栈

- **模型层**：OpenAI 兼容接口，支持 DeepSeek-Chat / Qwen-Plus / GLM-4-Flash 多模型对比
- **RAG 管线**：BGE-large-zh-v1.5 嵌入 → FAISS 向量索引 → Top-K 检索
- **评测体系**：LLM-as-judge（裁判模型评定正确/错误/拒答）
- **可视化**：Matplotlib 多维度图表（幻觉率 HR、事实准确率 FA、缓解增益 MG）

### 3.3 评测指标

| 指标 | 全称 | 含义 |
|------|------|------|
| HR | Hallucination Rate | 产生幻觉的回答占比（越低越好） |
| FA | Factual Accuracy | 完全正确的回答占比（越高越好） |
| RA | Refusal Accuracy | 正确拒答的占比 |
| MG | Mitigation Gain | 缓解增益 = (G1_HR - Gx_HR) / G1_HR × 100% |

## 四、项目结构

```
llm-hallucination/
├── run.py                    # 主入口：一键运行实验
├── requirements.txt          # Python 依赖
├── .env.example              # API 密钥配置模板
├── data/
│   ├── questions.json        # 20 道事实型测试题目
│   └── knowledge_base/       # 维基百科知识库数据
├── src/
│   ├── config.py             # 全局配置（模型、RAG 参数）
│   ├── models/
│   │   └── api.py            # OpenAI 兼容多模型调用接口
│   ├── experiments/
│   │   ├── prompts.py        # 5 组消融实验 Prompt 模板
│   │   └── runner.py         # 实验运行器
│   ├── rag/
│   │   ├── build_kb.py       # 知识库索引构建
│   │   └── retriever.py      # FAISS 检索器
│   ├── evaluation/
│   │   └── metrics.py        # LLM-as-judge 评测与指标计算
│   └── visualization/
│       └── plots.py          # 结果可视化
```

## 五、环境配置与运行

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env 文件，至少填入一个模型的 API key
```

支持的模型平台：
- DeepSeek：https://platform.deepseek.com
- 阿里云 DashScope（Qwen）：https://dashscope.aliyun.com
- 智谱 BigModel（GLM）：https://open.bigmodel.cn

### 5.3 运行实验

```bash
# 一键运行完整实验
python run.py

# 指定模型和实验组
python run.py --models deepseek-chat qwen-plus --groups G1 G2

# 仅运行 G1 基线
python run.py --groups G1 --skip-kb

# 仅评估已有结果
python run.py --eval-only data/results/results_xxx.json
```

## 六、实验结果（G1 基线）

使用 DeepSeek-Chat 对 20 道事实型题目进行 G1 基线测试。基线幻觉率反映了模型在无外部知识辅助情况下的表现，为后续 G2~G5 缓解方案的对比提供参照基准。

> 基线幻觉率高并非缺陷——恰好说明存在真实的幻觉问题需要解决，为缓解方案的有效性验证提供了充分的实验空间。

## 七、重点难点

1. **幻觉的精准判定**：简单关键词匹配无法识别中文同义表达和数字格式差异，本项目采用 LLM-as-judge 裁判模型进行语义级判断
2. **RAG 检索质量**：知识库覆盖度和检索精度直接影响 G2~G5 实验效果，需持续扩充知识库
3. **多模型 API 稳定性**：不同厂商的 API 响应延迟和限流策略存在差异，实验运行器内置了频率控制机制
4. **实验复现性**：通过固定温度参数（T=0.1）和标准化 prompt 模板控制变量

## 八、后续计划

- [ ] 扩充测试题目集，覆盖更多题型（推理题、开放题）
- [ ] 引入更多缓解策略对比（Chain-of-Verification、RARR 等）
- [ ] 接入 Qwen-Plus 和 GLM-4-Flash 完成多模型对比
- [ ] 使用人工标注验证 LLM-as-judge 的评测准确率
