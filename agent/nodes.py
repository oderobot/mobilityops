"""agent/nodes.py —— 图节点逻辑（纯函数，零框架依赖，可独立单测）

设计要点：
1. 节点 = (MobilityState) -> 增量状态 的纯函数。LangGraph 只负责接线，
   全部业务逻辑在这里，不需要起图、不需要 API key 就能测试。
2. call_llm 依赖注入：签名 (messages, tools|None) -> assistant 消息 dict。
   测试注入 FakeLLM（剧本式响应），生产注入真客户端适配器（见 graph.py）。
3. 降级复用 agent 节点：步数耗尽时 agent_node 自动改为"不带 tools 的总结调用"，
   拓扑不变、无额外节点 —— raw_loop 修正点 3 在图里的优雅实现。
4. 工具执行只经过 tools.registry（窄腰），节点不直接 import 任何具体工具。

assistant 消息的内部统一格式（dict）：
    {"role": "assistant", "content": str|None,
     "tool_calls": [{"id": str, "name": str, "arguments": str}]}  # 可选
"""

from typing import Callable

from agent.state import MobilityState, make_observation, make_step_record
from tools import registry

# call_llm 的契约类型
CallLLM = Callable[[list[dict], list[dict] | None], dict]


def make_nodes(call_llm: CallLLM) -> dict[str, Callable]:
    """节点工厂：注入 LLM 可调用对象，返回三个节点函数。"""

    def agent_node(state: MobilityState) -> dict:
        """Thought 节点：调 LLM 决定下一步。

        两种模式：
        - 正常：携带工具 schema，模型可自由决定调工具或直接回答；
        - 降级（step_count 已达上限）：不带 tools + 追加总结指令，
          强制模型基于已有观测收尾（raw_loop 修正点 3 的图内实现）。
        """
        step = state["step_count"] + 1
        messages = list(state["messages"])
        degraded = step > state["max_steps"]

        if degraded:
            messages.append({
                "role": "user",
                "content": "已达到最大工具调用次数。请基于目前已收集的信息直接总结回答，"
                           "并明确说明哪些部分未能完成。",
            })

        try:
            response = call_llm(messages, None if degraded else registry.get_openai_tools())
        except Exception as exc:  # LLM 服务失败：结构化失败，不崩溃
            return {
                "step_count": step,
                "status": "failed",
                "error": f"LLM 调用失败：{exc}",
                "executed_steps": [make_step_record(step, "llm_call", "llm", f"调用失败：{exc}")],
            }

        record = make_step_record(
            step, "llm_call", "llm",
            f"请求 {len(response.get('tool_calls') or [])} 个工具调用"
            if response.get("tool_calls") else "直接给出文本回答",
        )
        update: dict = {
            "messages": [response],
            "step_count": step,
            "executed_steps": [record],
        }
        if degraded:
            update["status"] = "degraded"  # 标记降级结局，finalize 会保留
        return update

    def tools_node(state: MobilityState) -> dict:
        """Action + Observation 节点：执行工具、写回观测。

        所有调用经过 registry.execute（窄腰）：白名单/JSON 解析/参数校验/
        异常兜底全部在那里，本节点零防御代码 —— 复杂度各归其位。
        """
        last = state["messages"][-1]
        tool_calls = last.get("tool_calls") or []
        step = state["step_count"]

        tool_messages: list[dict] = []
        observations: list[dict] = []
        records: list[dict] = []

        for tc in tool_calls:
            result = registry.execute(tc["name"], tc["arguments"])
            obs_text = registry.to_observation(result)

            #【关键点】直接把大模型给的 id（tc["id"]）原封不动地抄过来！
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],  # OpenAI 协议硬要求：与 assistant 的调用配对
                "content": obs_text,
            })
            observations.append(make_observation(
                step, tc["name"], result.ok,
                summary=obs_text[:120],  # 压缩层只存摘要，细节留在 messages
                error_code=None if result.ok else result.error.code.value,
            ))
            records.append(make_step_record(
                step, "tool_call", tc["name"],
                f"{'成功' if result.ok else '失败: ' + result.error.code.value}",
            ))

        return {
            "messages": tool_messages,
            "observations": observations,
            "executed_steps": records,
        }

    def finalize_node(state: MobilityState) -> dict:
        """收尾节点：从最后一条 assistant 消息提取最终回答，确定结局状态。"""
        last = state["messages"][-1]
        status = state["status"]
        if status == "running":
            status = "success"  # 正常退出（模型不再调工具）
        # degraded / failed 已被 agent_node 标记，此处保留
        return {"status": status, "final_answer": last.get("content")}

    return {"agent": agent_node, "tools": tools_node, "finalize": finalize_node}


def route_after_agent(state: MobilityState) -> str:
    """条件边：agent 之后去哪。

    - LLM 调用已失败 → finalize（不折腾）
    - 模型请求了工具 → tools
    - 模型直接回答 → finalize
    """
    if state["status"] == "failed":
        return "finalize"
    last = state["messages"][-1]
    return "tools" if last.get("tool_calls") else "finalize"

