"""tests/test_orchestration.py —— 编排层测试：state + nodes（不起图、不需要 API key）

测试即规格（冻结决策 §0.3-5）：
  - HappyPath：FakeLLM 剧本驱动，手工跑完整 ReAct 回路
  - FailurePath：LLM 异常、幻觉工具名、步数耗尽降级
  - 核心不变量：tool_call_id 配对、双层上下文投影、节点永不抛异常

设计前提：节点是纯函数（nodes.py），LangGraph 只是接线（graph.py），
所以这里直接手工驱动 agent→tools→agent→finalize 验证全部业务逻辑。
graph.py 的组装正确性由本地 langgraph 环境下的冒烟测试验证。

运行方式：从项目根目录执行  pytest tests/ -v
"""

from agent.nodes import make_nodes, route_after_agent
from agent.state import (
    append_list,
    append_messages,
    initial_state,
    keep_max,
)


# ─────────────────────────────────────────────
# FakeLLM：剧本式 LLM，测试的依赖注入替身
# ─────────────────────────────────────────────

class FakeLLM:
    """按剧本依次返回响应；剧本项是 Exception 则抛出。记录每次调用便于断言。"""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[tuple] = []  # (messages, tools)

    def __call__(self, messages: list, tools: list | None) -> dict:
        self.calls.append((list(messages), tools))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _tool_call_msg(name: str, args: str, call_id: str = "call_1") -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "name": name, "arguments": args}]}


def _text_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _apply(state: dict, update: dict) -> dict:
    """手工模拟 LangGraph 的 reducer 合并（与图运行语义一致）。"""
    merged = dict(state)
    for k, v in update.items():
        if k == "messages":
            merged[k] = append_messages(merged.get(k), v)
        elif k in ("observations", "executed_steps"):
            merged[k] = append_list(merged.get(k), v)
        elif k == "step_count":
            merged[k] = keep_max(merged.get(k, 0), v)
        else:
            merged[k] = v
    return merged


# ─────────────────────────────────────────────
# state.py 纯函数
# ─────────────────────────────────────────────

def test_initial_state_fields():
    s = initial_state("诊断早高峰", max_steps=8)
    assert s["status"] == "running" and s["step_count"] == 0
    assert s["max_steps"] == 8 and s["final_answer"] is None
    assert len(s["messages"]) == 2  # system + user
    assert s["messages"][1]["content"] == "诊断早高峰"


def test_reducers():
    assert append_messages([1], [2, 3]) == [1, 2, 3]
    assert append_list(None, [1]) == [1]
    assert keep_max(3, 5) == 5 and keep_max(5, 3) == 5  # step_count 只增不减


# ─────────────────────────────────────────────
# HappyPath：完整 ReAct 回路
# ─────────────────────────────────────────────

def test_full_react_loop_happy_path():
    """剧本：第 1 轮请求工具 → 执行成功 → 第 2 轮直接回答。验证全链路状态。"""
    llm = FakeLLM([
        _tool_call_msg("predict_traffic_speed", '{"region_ids": [3], "time_slice": 100}'),
        _text_msg("3 号区域预测车速 25.4 km/h，属于拥堵。"),
    ])
    nodes = make_nodes(llm)
    state = initial_state("查 3 号区域车速", max_steps=5)

    # ── 第 1 轮：agent → route → tools ──
    state = _apply(state, nodes["agent"](state))
    assert state["step_count"] == 1
    assert route_after_agent(state) == "tools"

    state = _apply(state, nodes["tools"](state))
    assert len(state["observations"]) == 1
    assert state["observations"][0]["tool"] == "predict_traffic_speed"
    assert state["observations"][0]["ok"] is True

    # ── 第 2 轮：agent → route → finalize ──
    state = _apply(state, nodes["agent"](state))
    assert route_after_agent(state) == "finalize"

    state = _apply(state, nodes["finalize"](state))
    assert state["status"] == "success"
    assert "25.4" in state["final_answer"]

    # 决策轨迹完整：llm_call → tool_call → llm_call
    kinds = [r["kind"] for r in state["executed_steps"]]
    assert kinds == ["llm_call", "tool_call", "llm_call"]


def test_llm_receives_tool_schemas_and_history():
    """验证：agent 节点把 registry 的 schema 和完整历史（含观测）交给 LLM。"""
    llm = FakeLLM([
        _tool_call_msg("search_project_docs", '{"query": "MAE 口径"}'),
        _text_msg("文档说 MAE 是 12 步均值。"),
    ])
    nodes = make_nodes(llm)
    state = initial_state("MAE 口径是什么", max_steps=5)
    state = _apply(state, nodes["agent"](state))
    state = _apply(state, nodes["tools"](state))
    state = _apply(state, nodes["agent"](state))

    first_tools = llm.calls[0][1]
    assert len(first_tools) == 6  # registry 全部工具
    assert first_tools[0]["function"]["name"] == "predict_traffic_speed"
    # 第 2 次调用时，历史中包含 tool 角色的观测消息
    second_messages = llm.calls[1][0]
    assert any(m["role"] == "tool" for m in second_messages)


# ─────────────────────────────────────────────
# FailurePath
# ─────────────────────────────────────────────

def test_llm_exception_marks_failed_and_routes_to_finalize():
    """LLM 服务异常 → status=failed + 错误信息，条件边直接收尾，不崩溃。"""
    llm = FakeLLM([ConnectionError("API timeout")])
    nodes = make_nodes(llm)
    state = initial_state("任意目标")
    state = _apply(state, nodes["agent"](state))
    assert state["status"] == "failed"
    assert "API timeout" in state["error"]
    assert route_after_agent(state) == "finalize"


def test_hallucinated_tool_becomes_observation():
    """幻觉工具名：窄腰拦截，失败观测携带 error_code 和 retryable=false。"""
    llm = FakeLLM([_tool_call_msg("fly_to_moon", "{}"), _text_msg("工具不存在，无法完成。")])
    nodes = make_nodes(llm)
    state = initial_state("任意目标")
    state = _apply(state, nodes["agent"](state))
    state = _apply(state, nodes["tools"](state))

    obs = state["observations"][0]
    assert obs["ok"] is False and obs["error_code"] == "INVALID_PARAM"
    tool_msg = state["messages"][-1]
    assert "retryable=false" in tool_msg["content"]


def test_degraded_mode_when_max_steps_exceeded():
    """步数耗尽：agent 节点自动降级为不带 tools 的总结调用，status=degraded。"""
    llm = FakeLLM([_text_msg("基于已有信息总结：……")])
    nodes = make_nodes(llm)
    state = initial_state("任意目标", max_steps=2)
    state["step_count"] = 2  # 模拟已耗尽

    state = _apply(state, nodes["agent"](state))
    assert llm.calls[0][1] is None  # 降级调用不携带 tools
    assert "已达到最大工具调用次数" in llm.calls[0][0][-1]["content"]
    assert state["status"] == "degraded"

    state = _apply(state, nodes["finalize"](state))
    assert state["status"] == "degraded"  # finalize 保留降级标记
    assert "总结" in state["final_answer"]


# ─────────────────────────────────────────────
# 核心不变量
# ─────────────────────────────────────────────

def test_invariant_tool_call_id_pairing():
    """不变量 1：tool 消息的 tool_call_id 必须与 assistant 的调用严格配对（协议硬要求）。"""
    llm = FakeLLM([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_A", "name": "predict_traffic_speed", "arguments": '{"region_ids": [1], "time_slice": 0}'},
            {"id": "call_B", "name": "search_project_docs", "arguments": '{"query": "实验口径"}'},
        ]},
        _text_msg("完成"),
    ])
    nodes = make_nodes(llm)
    state = initial_state("多工具", max_steps=5)
    state = _apply(state, nodes["agent"](state))
    state = _apply(state, nodes["tools"](state))

    tool_msgs = [m for m in state["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_A", "call_B"]


def test_invariant_dual_projection():
    """不变量 2：双层上下文投影 —— messages 存细节，observations 只存压缩摘要。"""
    llm = FakeLLM([
        _tool_call_msg("predict_traffic_speed", '{"region_ids": [3], "time_slice": 100}'),
        _text_msg("done"),
    ])
    nodes = make_nodes(llm)
    state = initial_state("投影检查")
    state = _apply(state, nodes["agent"](state))
    state = _apply(state, nodes["tools"](state))

    tool_msg = [m for m in state["messages"] if m["role"] == "tool"][0]
    assert len(tool_msg["content"]) > 50  # 细节层：完整 JSON
    obs = state["observations"][0]
    assert len(obs["summary"]) <= 120    # 压缩层：摘要上限
    assert set(obs) <= {"step", "tool", "ok", "summary", "error_code"}


def test_invariant_nodes_never_raise():
    """不变量 3：节点对恶意状态永不抛异常（与窄腰契约一致）。"""
    llm = FakeLLM([_tool_call_msg("predict_traffic_speed", "not json")])  # 非法 JSON
    nodes = make_nodes(llm)
    state = initial_state("恶意输入")
    state = _apply(state, nodes["agent"](state))
    state = _apply(state, nodes["tools"](state))  # 窄腰拦下 LLM_INVALID_OUTPUT
    assert state["observations"][0]["error_code"] == "LLM_INVALID_OUTPUT"
