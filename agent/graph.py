"""agent/graph.py —— LangGraph 组装层（唯一的框架依赖点）

设计要点：
1. 本模块是项目中唯一 import langgraph 的地方 —— 框架被隔离在组装层，
   业务逻辑（nodes.py / state.py / registry）全部可脱离框架测试。
2. 图拓扑就是 raw_loop 的形式化：
       START → agent ──(条件边 route_after_agent)──→ tools → agent（回边）
                    └──────────→ finalize → END
   与 raw_loop 的 for 循环一一对应：agent=Thought，tools=Action+Observation。
3. checkpointer 与 HITL：传入 checkpointer 即获得状态持久化/断点恢复；
   hitl=True 时在 tools 节点前 interrupt（人工审批工具调用后再执行）。
4. run() 是对外唯一入口：注入 LLM 客户端，返回结构化结局 dict。

本地运行依赖：pip install langgraph（已加入 requirements.txt）
"""

from agent.nodes import make_nodes, route_after_agent
from agent.state import MobilityState, initial_state


def make_openai_call_llm(client, model: str = "gpt-4o"):
    """把真 OpenAI 客户端适配成 call_llm 契约。

    适配器职责：SDK 的 Message 对象 → 节点内部统一的 dict 格式。
    测试用 FakeLLM 时不需要本函数 —— 这就是依赖注入的价值。
    """
    def call_llm(messages: list[dict], tools: list[dict] | None) -> dict:
        kwargs: dict = {"model": model, "messages": messages}
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        out: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            out["tool_calls"] = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]
        return out
    return call_llm


def build_graph(call_llm, checkpointer=None, hitl: bool = False):
    """组装 StateGraph。

    参数：
        call_llm:    见 nodes.CallLLM 契约（测试注入 FakeLLM，生产用 make_openai_call_llm）
        checkpointer: LangGraph 检查点（如 SqliteSaver），给了就有断点恢复能力
        hitl:        True 时在每次工具执行前中断，等待人工批准后继续
    """
    from langgraph.graph import END, START, StateGraph  # 延迟导入：框架隔离在组装层

    nodes = make_nodes(call_llm)
    builder = StateGraph(MobilityState)

    builder.add_node("agent", nodes["agent"])
    builder.add_node("tools", nodes["tools"])
    builder.add_node("finalize", nodes["finalize"])

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent,
                                  {"tools": "tools", "finalize": "finalize"})
    builder.add_edge("tools", "agent")      # 回边 = raw_loop 的下一轮迭代
    builder.add_edge("finalize", END)

    # HITL：interrupt_before 让图在 tools 节点前暂停，
    # 外部审批后用同一 thread_id 调 invoke(None, config) 恢复
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"] if hitl else None,
    )


def run(goal: str, call_llm, max_steps: int = 10, checkpointer=None, hitl: bool = False) -> dict:
    """对外唯一入口：跑一个诊断任务，返回结构化结局。"""
    graph = build_graph(call_llm, checkpointer=checkpointer, hitl=hitl)
    config = {"configurable": {"thread_id": "mobilityops-demo"}} if checkpointer else None
    final = graph.invoke(initial_state(goal, max_steps), config=config)
    return {
        "status": final["status"],
        "answer": final.get("final_answer"),
        "steps_used": final["step_count"],
        "observations": final["observations"],
        "executed_steps": final["executed_steps"],
    }

# 如何实现 Agent 框架与业务逻辑的解耦？
#
#         回答：采用适配器模式与依赖注入。通过定义统一的契约（如 CallLLM 接口），将 OpenAI SDK 包装在最外层的组装层（graph.py），核心节点和状态对具体的大模型 SDK 零感知，从而保证了极高的可测性和可迁移性。
#
#     生产级 Agent 如何支持中断与人工审批（HITL）？
#
#         回答：利用图框架的中断机制（如 LangGraph 的 interrupt_before）。在涉及写操作或高危动作的节点前设置断点，将状态持久化到 Checkpointer 中，等待人工介入审核批准后，再带上相同的 thread_id 恢复执行。
#
#     状态机拓扑设计与原生循环（While Loop）的映射关系：
#
#         回答：图中的条件边和回边（tools 指回 agent）本质上就是传统代码中 while True 循环的形式化表达。用状态图的好处在于，每一个节点的输入输出、历史状态追踪、异常中断和可视化都被框架标准化了。