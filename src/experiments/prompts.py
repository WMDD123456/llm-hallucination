"""
5 组消融实验的 prompt 模板

G1: 原生生成（Baseline）
G2: RAG 检索增强
G3: RAG + 强制证据引用
G4: RAG + Self-check 二次校验
G5: 完整方案（RAG + 证据引用 + Self-check）
"""


def build_g1_prompt(question: str) -> str:
    """G1: 直接问答"""
    return question


def build_g2_prompt(question: str, contexts: list[str]) -> str:
    """G2: RAG 检索增强 — 将检索到的段落注入 prompt"""
    ctx_text = "\n\n".join(f"【参考资料 {i+1}】{c}" for i, c in enumerate(contexts))
    return f"""请根据以下参考资料回答问题。

{ctx_text}

问题：{question}
答案："""


def build_g3_prompt(question: str, contexts: list[str]) -> str:
    """G3: RAG + 强制证据引用 — 要求回答中标注引用来源"""
    ctx_text = "\n\n".join(f"[来源{i+1}] {c}" for i, c in enumerate(contexts))
    return f"""请根据以下参考资料回答问题。你的回答中必须标注引用来源（用 [来源N] 格式）。

{ctx_text}

问题：{question}
请用以下格式回答：
答案：（你的答案）
依据：（引用 [来源N] 说明）"""


def build_g4_prompt(question: str, contexts: list[str]) -> str:
    """G4: RAG + Self-check — 先生成答案，再自查"""
    ctx_text = "\n\n".join(f"【参考资料 {i+1}】{c}" for i, c in enumerate(contexts))
    return f"""请根据以下参考资料回答问题，然后检查你的答案的准确性。

{ctx_text}

问题：{question}

请先给出答案，然后进行自我检查，格式如下：
答案：（你的答案）
自查：（逐条核实答案中的事实是否与参考资料一致，如发现不一致请指出）"""


def build_g5_prompt(question: str, contexts: list[str]) -> str:
    """G5: 完整方案 — RAG + 引用 + Self-check"""
    ctx_text = "\n\n".join(f"[来源{i+1}] {c}" for i, c in enumerate(contexts))
    return f"""请根据以下参考资料回答问题，标注引用来源，并进行自我检查。

{ctx_text}

问题：{question}

请按以下格式回答：
答案：（你的答案）
依据：（引用来源编号说明）
自查：（核实答案是否与参考资料一致）"""


PROMPT_BUILDERS = {
    "G1": build_g1_prompt,
    "G2": build_g2_prompt,
    "G3": build_g3_prompt,
    "G4": build_g4_prompt,
    "G5": build_g5_prompt,
}

GROUP_LABELS = {
    "G1": "Baseline（原生生成）",
    "G2": "RAG 检索增强",
    "G3": "RAG + 强制证据引用",
    "G4": "RAG + Self-check",
    "G5": "完整方案（RAG + 引用 + Self-check）",
}
