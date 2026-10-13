"""tools/search_project_docs.py —— Tool 6：项目文档检索（RAG，轻量确定性实现）

干什么：检索项目语料（开发文档/manifest/手册等），返回带引用的片段。
实现口径：BM25（纯 Python）+ 哈希向量（确定性 mock embedding）双路召回，
    RRF 融合（k=60 标准常数）。零重依赖、同输入必同输出。
    TODO(论文接入)：哈希向量 → bge-small-zh，纯 Python BM25 → rank_bm25/FAISS。
检索失败纪律（§5.6）：无命中显式返回空 + 说明，不让 LLM 自己编。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .base import ErrorCode, ToolResult
from .predict_traffic_speed import _format_validation_error

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50   # §7 RAG 设计口径
EMB_DIM = 128                          # mock 哈希向量维度
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


class SearchDocsInput(BaseModel):
    query: str = Field(min_length=1, description="检索问题。严禁传空串。")
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="返回条数。严禁 >20：过多上下文稀释关键信息且浪费 token。",
    )


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunk(text: str, source: str) -> list[dict]:
    """递归字符切分的简化版：滑窗 + overlap（标题层级优先为 TODO）。"""
    chunks, step = [], CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, max(len(text), 1), step):
        piece = text[i:i + CHUNK_SIZE].strip()
        if piece:
            chunks.append({"content": piece, "source": source})
    return chunks


def _load_corpus(docs_dir: Path) -> list[dict]:
    corpus = []
    for p in sorted(docs_dir.rglob("*")):
        if p.suffix in (".md", ".txt", ".yaml", ".yml"):
            corpus.extend(_chunk(p.read_text(encoding="utf-8", errors="ignore"), p.name))
    return corpus


def _hash_embed(tokens: list[str]) -> list[float]:
    """确定性 mock embedding：token 哈希撒桶。TODO(论文接入): bge-small-zh。"""
    vec = [0.0] * EMB_DIM
    for t in tokens:
        h = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big")
        vec[h % EMB_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _bm25_rank(query_tokens: list[str], corpus_tokens: list[list[str]]) -> list[float]:
    """标准 BM25：k1=1.5，b=0.75。词频饱和 + 文档长度归一化。"""
    k1, b = 1.5, 0.75
    n = len(corpus_tokens)
    avgdl = sum(len(c) for c in corpus_tokens) / max(n, 1)
    df = Counter()
    for c in corpus_tokens:
        for t in set(c):
            df[t] += 1
    scores = []
    for c in corpus_tokens:
        tf = Counter(c)
        score = 0.0
        for t in query_tokens:
            if tf[t] == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(c) / avgdl))
        scores.append(score)
    return scores


def _rrf_merge(bm25_scores: list[float], dense_scores: list[float]) -> list[tuple[int, float]]:
    """RRF 融合：score = Σ 1/(k + rank)，两路各算一个倒数排名分。"""
    def ranks(scores):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return {idx: r for r, idx in enumerate(order)}
    rb, rd = ranks(bm25_scores), ranks(dense_scores)
    fused = [(i, 1 / (RRF_K + rb[i]) + 1 / (RRF_K + rd[i])) for i in range(len(bm25_scores))]
    return sorted(fused, key=lambda x: -x[1])


def search_project_docs(params: dict, docs_dir: Path | None = None) -> ToolResult:
    start = time.perf_counter()
    try:
        inp = SearchDocsInput.model_validate(params)
    except ValidationError as exc:
        return ToolResult.fail(ErrorCode.INVALID_PARAM, _format_validation_error(exc))

    docs_dir = docs_dir or DOCS_DIR
    if not docs_dir.exists():
        return ToolResult.fail(ErrorCode.DATA_MISSING, f"语料目录不存在：{docs_dir}")
    corpus = _load_corpus(docs_dir)
    if not corpus:
        return ToolResult.fail(ErrorCode.DATA_MISSING, "语料目录为空，请先向 data/docs/ 放入文档")

    q_tokens = _tokenize(inp.query)
    corpus_tokens = [_tokenize(c["content"]) for c in corpus]

    bm25_scores = _bm25_rank(q_tokens, corpus_tokens)
    q_vec = _hash_embed(q_tokens)
    dense_scores = [_cosine(q_vec, _hash_embed(ct)) for ct in corpus_tokens]
    fused = _rrf_merge(bm25_scores, dense_scores)

    # 零命中纪律：没有任何关键词重叠 → 显式空 + 说明，不让 LLM 编
    if not any(bm25_scores):
        return ToolResult.success({
            "results": [],
            "note": "文档中未找到与查询关键词相关的内容；请换关键词或补充语料，不要编造引用。",
            "provenance": {"tool": "search_project_docs",
                           "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)},
        })

    results = [
        {"content": corpus[i]["content"], "source": corpus[i]["source"], "score": round(s, 6)}
        for i, s in fused[: inp.top_k]
    ]
    return ToolResult.success({
        "results": results,
        "note": "结果按 BM25+哈希向量 RRF 融合排序；引用时请附 source 字段",
        "provenance": {"tool": "search_project_docs", "implementation": "bm25+hash-vector（mock embedding）",
                       "elapsed_ms": round((time.perf_counter() - start) * 1000, 2)},
    })


# 考点 1：在工业级 Agent 的 RAG 系统中，为什么要采用 BM25 与稠密向量（Dense Vector）相结合的双路召回，并用 RRF 进行融合？
#
#     满分回答：
#
#         “因为单纯的稠密向量检索（Embedding）虽然擅长捕捉语义相似度，但对于专有名词、精确代码片段、专业 ID（如 scenario_id）的检索常常会‘失明’；而传统的 BM25 稀疏检索则刚好相反，擅长精准字词匹配却不懂深层语义。通过双路召回互补，再利用 RRF（倒数排名融合） 算法在不依赖复杂归一化权重的情况下将两路结果安全融合，能够兼顾‘语义理解’与‘精准命中’，这在工业级 RAG 中是最稳健的工程实践。”
#
# 考点 2：当 RAG 系统检索不到相关文档时，如何防止大模型产生“检索幻觉（Retrieval Hallucination）”？
#
#     满分回答：
#
#         “必须在底层工具的输出契约中引入**‘零命中硬门禁（Zero-Hit Gate）’**。正如 Tool 6 源码所示：当检索评分为零或没有命中任何切片时，工具主动返回空列表，并在返回结果的 note 字段中显式写入硬编码警告。同时，在 Prompt 中强制约束大模型：‘若检索结果为空，严禁自由发挥，必须直接告知用户未找到相关内部文档’。这从系统工程和提示词两端彻底封死了大模型胡乱编造引用的可能性。”