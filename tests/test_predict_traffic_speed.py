"""tests/test_predict_traffic_speed.py —— Tool 1 测试：HappyPath / FailurePath / 核心不变量

测试即规格（冻结决策 §0.3-5）：
  - HappyPath：合法参数下的正常行为
  - FailurePath：非法输入必须得到结构化错误，且 retryable 符合 §9 分类表
  - 核心不变量：输出长度、速度值域、确定性、region 顺序无关、质量口径存在

运行方式：从项目根目录执行  pytest tests/ -v
"""

from tools.predict_traffic_speed import (
    MAX_HORIZON_STEPS,
    N_REGIONS,
    SPEED_CAP_KMH,
    predict_traffic_speed,
)


# ──────────────────────── HappyPath ────────────────────────

class TestHappyPath:
    def test_default_params_single_region(self):
        """最简合法调用：单区域 + 全默认参数。"""
        r = predict_traffic_speed({"region_ids": [3]})
        assert r.ok and r.error is None
        preds = r.data["predictions"]
        assert list(preds.keys()) == ["3"]
        assert len(preds["3"]) == MAX_HORIZON_STEPS  # 默认预测 12 步

    def test_multi_region_and_custom_horizon(self):
        r = predict_traffic_speed({"region_ids": [0, 5, 155], "horizon_steps": 3})
        assert r.ok
        assert set(r.data["predictions"].keys()) == {"0", "5", "155"}
        assert all(len(v) == 3 for v in r.data["predictions"].values())

    def test_output_carries_quality_reference_and_provenance(self):
        """结果必须带质量口径与溯源信息，否则 LLM 无法正确引用。"""
        r = predict_traffic_speed({"region_ids": [0]})
        qr = r.data["quality_reference"]
        assert qr["mae_test_mean"] == 2.78 and qr["unit"] == "km/h"
        assert qr["status"] == "verified"
        assert "置信度" in qr["note"]  # 防止被当成单次预测置信度的显式声明
        assert r.data["source"] == "mock"
        assert r.data["provenance"]["tool"] == "predict_traffic_speed"


# ──────────────────────── FailurePath ────────────────────────

class TestFailurePath:
    def test_region_out_of_range(self):
        r = predict_traffic_speed({"region_ids": [156]})  # 合法范围 0–155
        assert not r.ok
        assert r.error.code.value == "INVALID_PARAM"
        assert r.error.retryable is False  # 参数错误不应自动重试（§9）

    def test_negative_region(self):
        r = predict_traffic_speed({"region_ids": [-1]})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_empty_region_list(self):
        r = predict_traffic_speed({"region_ids": []})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_horizon_beyond_training_window(self):
        """horizon > 12 是负向约束：超出训练 output_window 属于不可靠外推。"""
        r = predict_traffic_speed({"region_ids": [0], "horizon_steps": 13})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_negative_time_slice(self):
        r = predict_traffic_speed({"region_ids": [0], "time_slice": -1})
        assert not r.ok and r.error.code.value == "INVALID_PARAM"

    def test_manifest_missing_returns_data_missing(self, monkeypatch):
        """manifest 缺失 → DATA_MISSING（可重试），严禁在缺口径时返回预测。"""
        import tools.predict_traffic_speed as mod

        monkeypatch.setattr(mod, "load_manifest", lambda: None)
        r = mod.predict_traffic_speed({"region_ids": [0]})
        assert not r.ok
        assert r.error.code.value == "DATA_MISSING"
        assert r.error.retryable is True  # 数据可能稍后就绪（§9）


# ──────────────────────── 核心不变量 ────────────────────────

class TestInvariants:
    def test_output_length_equals_horizon(self):
        for h in (1, 6, 12):
            r = predict_traffic_speed({"region_ids": [1, 2], "horizon_steps": h})
            assert all(len(v) == h for v in r.data["predictions"].values())

    def test_speed_within_physical_cap(self):
        """所有预测速度必须落在 [0, SPEED_CAP]（论文口径 86.43 km/h）。"""
        r = predict_traffic_speed({"region_ids": list(range(N_REGIONS)), "horizon_steps": 12})
        assert r.ok
        for series in r.data["predictions"].values():
            assert all(0.0 <= v <= SPEED_CAP_KMH for v in series)

    def test_determinism(self):
        """mock 铁律：同输入必得同输出（含 time_slice 维度）。"""
        params = {"region_ids": [7, 42], "horizon_steps": 12, "time_slice": 100}
        r1 = predict_traffic_speed(params)
        r2 = predict_traffic_speed(dict(params))
        assert r1.data["predictions"] == r2.data["predictions"]

    def test_region_order_independence(self):
        """region_ids 打乱顺序，每个区域的预测值不变（per-region 独立）。"""
        r1 = predict_traffic_speed({"region_ids": [3, 7], "time_slice": 5})
        r2 = predict_traffic_speed({"region_ids": [7, 3], "time_slice": 5})
        assert r1.data["predictions"]["3"] == r2.data["predictions"]["3"]
        assert r1.data["predictions"]["7"] == r2.data["predictions"]["7"]

    def test_time_slice_changes_prediction(self):
        """不同时间片应给出不同预测（时间锚点确实生效，不是摆设）。"""
        r1 = predict_traffic_speed({"region_ids": [0], "time_slice": 0})
        r2 = predict_traffic_speed({"region_ids": [0], "time_slice": 1})
        assert r1.data["predictions"]["0"] != r2.data["predictions"]["0"]
