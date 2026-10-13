"""agent/state.py —— MobilityState：编排层的共享状态（纯数据，零框架依赖）

设计要点（面试考点）：
1. messages 与 observations 分层（上下文工程 §4.1）：
   - messages：LLM 的完整上下文，保留工具输出细节（LLM 推理需要）；
   - observations：压缩摘要层（工具名/成败/error_code/关键信息），
     供 Re-plan、复盘、UI 展示消费 —— 同一事实的两个投影。
2. reducer 自己实现（append_messages）：消息是普通 dict，追加语义足够；
   state.py 不 import 任何 Agent 框架，可独立单测。
3. status 四态结构化结局（running/success/degraded/failed），
   与 raw_loop 的结构化返回一脉相承。

LangGraph 接线时，Annotated 的 reducer 被 StateGraph 用于合并节点返回值；
纯 Python 场景（单测）下直接调用 reducer 函数本身即可。
"""

from typing import Annotated, Any, Literal, TypedDict


# ─────────────────────────────────────────────
# Reducer：LangGraph 用它合并各节点返回的增量状态
# ─────────────────────────────────────────────

def append_messages(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """消息列表 reducer：追加语义。

    节点返回 {"messages": [新消息]} 时，LangGraph 调用本函数把新消息
    合并进历史，而不是整体覆盖 —— 这是状态图"增量更新"的核心机制。
    """
    return list(left or []) + list(right or [])


def append_list(left: list | None, right: list | None) -> list:
    """通用列表追加 reducer（observations / executed_steps 共用）。"""
    return list(left or []) + list(right or [])


def keep_max(left: int, right: int) -> int:
    """标量 reducer：取较大值。step_count 只增不减，防止节点返回旧值造成回退。"""
    return max(left or 0, right or 0)


# ─────────────────────────────────────────────
# 状态定义
# ─────────────────────────────────────────────
#degraded是降级状态
TaskStatus = Literal["running", "success", "degraded", "failed"]


class MobilityState(TypedDict, total=False):
    """编排层共享状态。total=False：节点只需返回它修改的字段。"""

    # ── 输入 ──
    goal: str                       # 用户原始目标（Re-plan 时对照）
    max_steps: int                  # 熔断上限

    # ── LLM 上下文（细节层）──
    messages: Annotated[list[dict], append_messages]

    # ── 压缩投影层（上下文工程 §4.1）──
    observations: Annotated[list[dict], append_list]     # 工具结果摘要
    executed_steps: Annotated[list[dict], append_list]   # 决策轨迹

    # ── 控制 ──
    step_count: Annotated[int, keep_max]  # LLM 调用次数（只增不减）
    status: TaskStatus
    final_answer: str | None
    error: str | None


# ─────────────────────────────────────────────
# 纯函数工具：初始化与记录（全部可独立单测）
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是 MobilityOps，一个网约车车队调度诊断助手。"
    "你可以调用工具预测车速、估计有效需求、优化再平衡、运行基线对比、检索项目文档。"
    "规则：1) 需要数据时必须调用工具，禁止编造数字；"
    "2) 工具失败时根据 error_code 和 retryable 决定重试或换方案；"
    "3) 引用论文指标时注明统计口径。"
)


def initial_state(goal: str, max_steps: int = 10) -> MobilityState:
    """构造初始状态。max_steps 默认 10：诊断链 predict→estimate→optimize 三步 × 2 纠错余量 + 余裕。"""
    return MobilityState(
        goal=goal,
        max_steps=max_steps,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ],
        observations=[],
        executed_steps=[],
        step_count=0,
        status="running",
        final_answer=None,
        error=None,
    )


def make_observation(step: int, tool_name: str, ok: bool, summary: str,
                     error_code: str | None = None) -> dict:
    """一条压缩观测。只保留 Re-plan/复盘需要的信息，不存完整 data。"""
    obs: dict[str, Any] = {"step": step, "tool": tool_name, "ok": ok, "summary": summary}
    if error_code is not None:
        obs["error_code"] = error_code
    return obs


def make_step_record(step: int, kind: Literal["llm_call", "tool_call"],
                     name: str, detail: str) -> dict:
    """一条决策轨迹记录。kind: llm_call（思考）或 tool_call（行动）。"""
    return {"step": step, "kind": kind, "name": name, "detail": detail}

# 考点 1：如何优雅地处理 Agent 的记忆与多轮对话？（上下文工程）
#
#     面试官想听什么：你不能把所有东西都无脑往大模型的 messages 里扔。
#
#     满分回答：正如代码中所体现的，生产级系统必须做分层设计。messages 留给 LLM 做严密推理，而 observations 和 executed_steps 用来做轻量级的结构化投影，专供后续的 Re-plan（自我反思）、复盘日志以及前端 UI 消费。
#
# 考点 2：并发或重试时，状态冲突（Race Condition）怎么解决？
#
#     面试官想听什么：多个节点同时修改状态或者发生重试时，怎么保证数据不乱？
#
#     满分回答：利用 Reducer（归约函数）。比如用 append_messages 做追加语义，用 keep_max 保证步数只增不减。这把状态合并的主动权交给了确定性的算法，而不是交给玄学的 LLM。
#
# 考点 3：如何降低 Agent 的单元测试成本？（可测试性 Design Patterns）
#
#     面试官想听什么：你的 Agent 每次测试都要花钱去调 GPT 吗？
#
#     满分回答：通过关注点分离（Separation of Concerns）。状态定义层（state.py）和纯函数工具（initial_state, make_observation）做到零框架依赖、零外部网络 IO。测试时完全可以在本地沙箱里用纯 Python 验证状态转换逻辑，做到“零 Token 消耗、秒级回归”。
#
# 考点 4：如何对 Agent 的执行生命周期进行收敛？
#
#     面试官想听什么：Agent 跑飞了或者陷入死循环怎么办？
#
#     满分回答：通过强类型的状态控制。代码中定义的 TaskStatus = Literal["running", "success", "degraded", "failed"] 配合 max_steps 熔断上限，给整个动态推理过程设置了硬性的安全边界，确保任务无论如何都能收敛到一个明确的终态。