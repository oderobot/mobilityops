"""tests/test_search_project_docs.py —— Tool 6：HappyPath / FailurePath / 检索纪律不变量"""

from pathlib import Path

from tools.search_project_docs import search_project_docs


class TestHappyPath:
    def test_hit_returns_with_source(self):
        r = search_project_docs({"query": "STGCN MAE 是多少"})
        assert r.ok and len(r.data["results"]) >= 1
        for item in r.data["results"]:
            assert item["source"] and item["score"] > 0  # 每条结果必须带出处

    def test_top_k_respected(self):
        r = search_project_docs({"query": "调度 基线 OSR", "top_k": 2})
        assert len(r.data["results"]) <= 2

    def test_determinism(self):
        r1 = search_project_docs({"query": "网约车 调度"})
        r2 = search_project_docs({"query": "网约车 调度"})
        assert r1.data["results"] == r2.data["results"]  # 检索也必须确定性


class TestFailurePath:
    def test_empty_query(self):
        r = search_project_docs({"query": ""})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_top_k_over_limit(self):
        r = search_project_docs({"query": "调度", "top_k": 21})  # 负向约束：严禁 >20
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_missing_corpus_dir(self, tmp_path):
        r = search_project_docs({"query": "调度"}, docs_dir=tmp_path / "ghost")
        assert not r.ok and r.error.code.value == "DATA_MISSING"

    def test_empty_corpus(self, tmp_path):
        r = search_project_docs({"query": "调度"}, docs_dir=tmp_path)
        assert not r.ok and r.error.code.value == "DATA_MISSING"


class TestInvariants:
    def test_zero_hit_returns_empty_with_note(self):
        """检索纪律：无命中显式空 + 说明，不返回低质量结果让 LLM 编。"""
        r = search_project_docs({"query": "zzzqqq xyzzy qwirk"})  # 纯乱码，零关键词重叠
        assert r.ok and r.data["results"] == []
        assert "未找到" in r.data["note"]

    def test_relevance_order(self):
        """含查询词的文档必须排在无关文档前。"""
        r = search_project_docs({"query": "MAPE 除零"})
        assert r.ok and r.data["results"]
        assert "MAPE" in r.data["results"][0]["content"]
