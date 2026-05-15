"""FAISS 检索器接口"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class Retriever:
    def __init__(self, index_dir: str | Path, embedding_model: str = "BAAI/bge-large-zh-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = FAISS.load_local(
            str(index_dir), self.embeddings, allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]


# 全局检索器实例（延迟初始化）
_retriever: Retriever | None = None


def get_retriever(index_dir: str | Path | None = None) -> Retriever:
    global _retriever
    if _retriever is None:
        if index_dir is None:
            from src.config import KB_INDEX_DIR
            index_dir = KB_INDEX_DIR
        _retriever = Retriever(index_dir)
    return _retriever
