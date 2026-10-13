"""tests/test_run_baseline.py —— Tool 4：HappyPath / FailurePath / 守恒+确定性+run_id 不变量"""

import json

import tools.run_baseline as mod
from tools.run_baseline import run_baseline


def _call(tmp_path, **over):
    """统一测试入口：落盘目录重定向到 tmp，避免污染真实结果池。"""
    mod.EXPERIMENTS_DIR = tmp_path
    params = {"strategy": "nearest", "fleet_size": 260,
              "time_slice": 0, "scenario_id": "test_sc"}
    params.update(over)
    return run_baseline(params)


class TestHappyPath:
    def test_all_four_strategies(self, tmp_path):
        for s in ("no_dispatch", "random", "nearest", "demand_only"):
            r = _call(tmp_path, strategy=s)
            assert r.ok, s
            assert 0.0 <= r.data["metrics"]["osr"] <= 1.0

    def test_result_persisted(self, tmp_path):
        r = _call(tmp_path)
        rec = json.loads((tmp_path / "test_sc" / f"{r.data['run_id']}.json").read_text())
        assert rec["metrics"] == r.data["metrics"]
        assert rec["params"]["strategy"] == "nearest"

    def test_metrics_note_declares_proxy(self, tmp_path):
        r = _call(tmp_path)
        assert "mock 派生" in r.data["metrics_note"]  # 派生指标必须显式标注


class TestFailurePath:
    def test_illegal_strategy(self, tmp_path):
        r = _call(tmp_path, strategy="congestion_aware")  # 负向约束：那是优化工具
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_fleet_size_out_of_range(self, tmp_path):
        r = _call(tmp_path, fleet_size=0)
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_run_id_conflict(self, tmp_path):
        r1 = _call(tmp_path, run_id="fixed_id")
        assert r1.ok
        r2 = _call(tmp_path, run_id="fixed_id", fleet_size=300)  # 同 id 不同参数
        assert not r2.ok and r2.error.code.value == "INVALID_PARAM"
        assert not r2.error.retryable

    def test_run_id_reuse_same_params_ok(self, tmp_path):
        r1 = _call(tmp_path, run_id="fixed_id")
        r2 = _call(tmp_path, run_id="fixed_id")  # 同 id 同参数 = 复现，放行
        assert r2.ok and r1.data["metrics"] == r2.data["metrics"]


class TestInvariants:
    def test_no_dispatch_moves_nothing(self, tmp_path):
        r = _call(tmp_path, strategy="no_dispatch")
        m = r.data["metrics"]
        assert m["total_moved"] == 0 and m["reposition_distance"] == 0.0

    def test_determinism_same_inputs(self, tmp_path):
        r1 = _call(tmp_path, strategy="random", time_slice=5)
        r2 = _call(tmp_path, strategy="random", time_slice=5)
        assert r1.data["metrics"] == r2.data["metrics"]  # random 也必须确定

    def test_run_id_identity(self, tmp_path):
        """同参数同 run_id，不同参数不同 run_id。"""
        a = _call(tmp_path)
        b = _call(tmp_path)
        c = _call(tmp_path, fleet_size=300)
        assert a.data["run_id"] == b.data["run_id"]
        assert a.data["run_id"] != c.data["run_id"]
