"""tests/test_estimate_effective_demand.py —— Tool 2：HappyPath / FailurePath / 物理单调性不变量

本工具的核心不变量（冻结决策 §0.3-5 指定）：物理单调性——
速度越高 → rho 越大 → 有效需求越小。这是论文因果链的代码化断言。
"""

from tools.estimate_effective_demand import (
    DEFAULT_BASE_DEMAND,
    MIN_CAPACITY_RATIO,
    capacity_ratio,
    effective_demand,
    estimate_effective_demand,
)


# ──────────────────────── HappyPath ────────────────────────

class TestHappyPath:
    def test_default_base_demand(self):
        """缺省 raw_demand 时用论文合成基准 0.5。"""
        r = estimate_effective_demand({"region_ids": [3], "time_slice": 0})
        assert r.ok
        d = r.data["capacity_detail"]["3"]
        assert d["raw_demand"] == DEFAULT_BASE_DEMAND
        # d_eff = raw / rho，三者必须自洽
        assert abs(r.data["effective_demand"]["3"] - d["raw_demand"] / d["rho"]) < 1e-3

    def test_custom_raw_demand(self):
        r = estimate_effective_demand({"region_ids": [0, 1], "raw_demand": {"0": 2.0}})
        assert r.ok
        assert r.data["capacity_detail"]["0"]["raw_demand"] == 2.0
        assert r.data["capacity_detail"]["1"]["raw_demand"] == DEFAULT_BASE_DEMAND

    def test_provenance_chain_includes_upstream(self):
        """溯源链不断：Tool 2 的 provenance 里能追到 Tool 1。"""
        r = estimate_effective_demand({"region_ids": [0]})
        assert r.data["provenance"]["upstream"]["tool"] == "predict_traffic_speed"


# ──────────────────────── FailurePath ────────────────────────

class TestFailurePath:
    def test_negative_demand_rejected(self):
        """需求为负 → INVALID_PARAM，不重试（负向约束的代码实现）。"""
        r = estimate_effective_demand({"region_ids": [0], "raw_demand": {"0": -1.0}})
        assert not r.ok
        assert r.error.code.value == "INVALID_PARAM" and r.error.retryable is False

    def test_empty_region_list(self):
        r = estimate_effective_demand({"region_ids": []})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_upstream_error_passthrough(self, monkeypatch):
        """冻结决策 §0.3-3：上游（Tool 1）失败原样透传，retryable 保留。"""
        import tools.estimate_effective_demand as mod
        from tools.base import ErrorCode, ToolResult

        monkeypatch.setattr(mod, "predict_traffic_speed",
                            lambda p: ToolResult.fail(ErrorCode.DATA_MISSING, "缓存缺失"))
        r = mod.estimate_effective_demand({"region_ids": [0]})
        assert not r.ok
        assert r.error.code.value == "DATA_MISSING"   # code 不被改写
        assert r.error.retryable is True              # retryable 不被吞掉
        assert r.error.message == "缓存缺失"


# ──────────────────── 核心不变量：物理单调性 ────────────────────

class TestPhysicalMonotonicity:
    """论文 ρ(v) 的已知跳点（忠实复现，见工具源码注释）：
    v=20 处 0.6→0.5（向下，论文系数瑕疵）；v=40 处 0.7→1.0（向上）。
    不变量改为：分段内严格单调 + 跳点位置/大小符合已记录口径。
    """
    KNOWN_JUMPS = {20.0: (0.6, 0.5), 40.0: (0.7, 1.0)}

    def test_rho_monotonic_within_segments(self):
        """每个分段内部 rho 单调不减。"""
        segments = [(0.0, 4.9), (5.0, 9.9), (10.0, 19.9), (20.0, 39.9), (40.0, 86.43)]
        for lo, hi in segments:
            vs = [lo + (hi - lo) * i / 20 for i in range(21)]
            rhos = [capacity_ratio(v) for v in vs]
            assert all(b >= a - 1e-12 for a, b in zip(rhos, rhos[1:])), f"分段 [{lo},{hi}] 不单调"

    def test_known_jump_points_documented(self):
        """跳点位置和大小与已记录口径一致——若论文公式被改动，此测试会报警。"""
        eps = 1e-9
        for v, (before, after) in self.KNOWN_JUMPS.items():
            assert abs(capacity_ratio(v - eps) - before) < 1e-6, f"v={v} 跳前值漂移"
            assert abs(capacity_ratio(v) - after) < 1e-6, f"v={v} 跳后值漂移"

    def test_rho_bounds(self):
        """畅通 rho=1，严重拥堵 rho=0.3，值域闭合。"""
        assert capacity_ratio(40.0) == 1.0 and capacity_ratio(86.43) == 1.0
        assert capacity_ratio(0.0) == MIN_CAPACITY_RATIO
        assert all(MIN_CAPACITY_RATIO <= capacity_ratio(v) <= 1.0
                   for v in (0, 3, 7, 15, 30, 40, 100))

    def test_effective_demand_monotonic_in_congestion(self):
        """同一原始需求下：越堵（v 越低）有效需求越大——论文核心假设。"""
        d_slow = effective_demand(1.0, speed=3.0)    # 严重拥堵
        d_mid = effective_demand(1.0, speed=20.0)
        d_fast = effective_demand(1.0, speed=50.0)   # 畅通
        assert d_slow > d_mid > d_fast

    def test_congestion_amplifies_demand(self):
        """拥堵时有效需求必须大于原始需求（rho<1 ⇒ 放大），畅通时相等。"""
        assert effective_demand(0.5, speed=3.0) > 0.5
        assert effective_demand(0.5, speed=50.0) == 0.5
