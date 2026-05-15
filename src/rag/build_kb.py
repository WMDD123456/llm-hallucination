"""从中文维基百科段落构建 FAISS 知识库索引"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def build_knowledge_base(
    data_file: str | Path,
    index_dir: str | Path,
    embedding_model: str = "BAAI/bge-large-zh-v1.5",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    从 JSON 文件构建 FAISS 索引。
    JSON 格式: [{"title": "...", "content": "..."}, ...]
    """
    data_file = Path(data_file)
    index_dir = Path(index_dir)

    with open(data_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []
    for item in raw_data:
        text = f"{item.get('title', '')}\n{item.get('content', '')}"
        documents.append(Document(page_content=text, metadata={"title": item.get("title", "")}))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir))

    print(f"知识库构建完成：{len(documents)} 篇文档 → {len(chunks)} 个片段")
    print(f"索引保存至：{index_dir}")
    return len(chunks)


if __name__ == "__main__":
    import sys
    from src.config import KB_DIR, KB_INDEX_DIR, RAG_CONFIG

    data_file = sys.argv[1] if len(sys.argv) > 1 else KB_DIR / "wiki_data.json"
    build_knowledge_base(
        data_file=data_file,
        index_dir=KB_INDEX_DIR,
        embedding_model=RAG_CONFIG["embedding_model"],
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
    )
