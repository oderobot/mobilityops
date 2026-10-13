"""tests/test_optimize_rebalancing.py —— Tool 3：HappyPath / FailurePath / 车辆守恒不变量"""

from tools.optimize_rebalancing import optimize_rebalancing
from tools.simulation import N_REGIONS


class TestHappyPath:
    def test_default_call(self):
        r = optimize_rebalancing({"fleet_size": 260, "time_slice": 0})
        assert r.ok
        assert r.data["solver_status"] == "FEASIBLE"
        assert 0.0 <= r.data["expected_osr"] <= 1.0
        assert r.data["total_moved_vehicles"] >= 0

    def test_provenance_chain(self):
        r = optimize_rebalancing({"fleet_size": 100})
        assert r.data["provenance"]["upstream"]["tool"] == "estimate_effective_demand"
        assert "启发式" in r.data["solver"]  # 显式声明无最优性保证


class TestFailurePath:
    def test_fleet_size_zero(self):
        r = optimize_rebalancing({"fleet_size": 0})
        assert not r.ok and r.error.code.value == "INVALID_PARAM" and not r.error.retryable

    def test_fleet_size_too_large(self):
        r = optimize_rebalancing({"fleet_size": 1001})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_upstream_error_passthrough(self, monkeypatch):
        import tools.optimize_rebalancing as mod
        from tools.base import ErrorCode, ToolResult

        monkeypatch.setattr(mod, "estimate_effective_demand",
                            lambda p: ToolResult.fail(ErrorCode.DATA_MISSING, "需求数据缺失"))
        r = mod.optimize_rebalancing({"fleet_size": 260})
        assert not r.ok and r.error.code.value == "DATA_MISSING" and r.error.retryable is True


class TestInvariants:
    def test_vehicle_conservation(self, monkeypatch):
        """核心不变量：调度只搬不造，总车辆数恒等于 fleet_size。"""
        import tools.optimize_rebalancing as mod
        from tools.simulation import init_supply

        captured = {}

        real_greedy = mod.greedy_rebalance
        def spy(supply, demand):
            out, plan, moved, dist = real_greedy(supply, demand)
            captured["before"], captured["after"] = sum(supply), sum(out)
            return out, plan, moved, dist
        monkeypatch.setattr(mod, "greedy_rebalance", spy)

        for fleet in (1, 260, 1000):
            optimize_rebalancing({"fleet_size": fleet, "time_slice": 3})
            assert captured["before"] == fleet == captured["after"]

    def test_determinism(self):
        r1 = optimize_rebalancing({"fleet_size": 260, "time_slice": 7})
        r2 = optimize_rebalancing({"fleet_size": 260, "time_slice": 7})
        assert r1.data["plan_summary"] == r2.data["plan_summary"]
        assert r1.data["expected_osr"] == r2.data["expected_osr"]
