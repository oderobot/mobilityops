"""tests/test_compare_experiments.py —— Tool 5：HappyPath / FailurePath / 不标最优不变量"""

import tools.run_baseline as rb_mod
from tools.compare_experiments import compare_experiments
from tools.run_baseline import run_baseline

INLINE = [
    {"name": "A_高OSR高成本", "params": {}, "metrics": {"osr": 0.92, "avg_wait_minutes": 1.2, "empty_mile_ratio": 0.4}},
    {"name": "B_均衡", "params": {}, "metrics": {"osr": 0.87, "avg_wait_minutes": 2.0, "empty_mile_ratio": 0.2}},
    {"name": "C_低成本低OSR", "params": {}, "metrics": {"osr": 0.72, "avg_wait_minutes": 4.2, "empty_mile_ratio": 0.0}},
]


class TestHappyPath:
    def test_inline_mode(self):
        r = compare_experiments({"inline_results": INLINE})
        assert r.ok and len(r.data["table"]) == 3

    def test_tradeoff_labeled_per_row(self):
        """A/C 各有优劣维度；B 全居中 → 显式标"均衡型"；不存在全局"最优项"。"""
        r = compare_experiments({"inline_results": INLINE})
        rows = {row["name"]: row for row in r.data["table"]}
        assert rows["A_高OSR高成本"]["strengths"] and rows["A_高OSR高成本"]["weaknesses"]
        assert rows["C_低成本低OSR"]["strengths"] and rows["C_低成本低OSR"]["weaknesses"]
        assert rows["B_均衡"]["profile"] == "均衡型（无极端维度）"
        assert "best" not in str(r.data.keys()).lower()

    def test_pareto_keeps_tradeoff_candidates(self):
        """A（OSR 最高）与 C（成本最低）互相不支配，都应进 Pareto 候选。"""
        r = compare_experiments({"inline_results": INLINE})
        names = set(r.data["pareto_candidates"])
        assert "A_高OSR高成本" in names and "C_低成本低OSR" in names

    def test_no_significance_without_seed_data(self):
        r = compare_experiments({"inline_results": INLINE})
        assert r.data["significance"] is None  # 纪律：无 seed 数据不算 p 值

    def test_reference_mode_reads_disk(self, tmp_path):
        rb_mod.EXPERIMENTS_DIR = tmp_path
        import tools.compare_experiments as cmp_mod
        cmp_mod.EXPERIMENTS_DIR = tmp_path
        run = run_baseline({"strategy": "nearest", "fleet_size": 260,
                            "time_slice": 0, "scenario_id": "sc1"})
        r = compare_experiments({"experiment_ids": [{"scenario_id": "sc1",
                                                     "run_id": run.data["run_id"]}]})
        assert r.ok and r.data["table"][0]["metrics"]["osr"] > 0


class TestFailurePath:
    def test_both_modes_rejected(self):
        r = compare_experiments({"experiment_ids": [{"scenario_id": "a", "run_id": "b"}],
                                 "inline_results": INLINE})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_neither_mode_rejected(self):
        r = compare_experiments({})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_unknown_metric_rejected(self):
        r = compare_experiments({"inline_results": INLINE, "metrics": ["qps"]})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_missing_experiment_data(self, tmp_path):
        import tools.compare_experiments as cmp_mod
        cmp_mod.EXPERIMENTS_DIR = tmp_path
        r = compare_experiments({"experiment_ids": [{"scenario_id": "ghost", "run_id": "none"}]})
        assert not r.ok and r.error.code.value == "DATA_MISSING"  # 严禁编造对比数据
