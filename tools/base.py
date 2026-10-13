"""tools/base.py —— 统一错误协议（冻结设计 §0.3-1 / §5.0 / §9）

干什么：
    定义所有 Tool 与 Agent 编排层之间的"错误语言"。
    任何 Tool 都不向 LLM 抛异常，只返回 ToolResult；
    出错时携带结构化 ToolError（code + message + retryable）。

为什么这样设计（面试点）：
    1. retryable 不允许各工具自行填写，而是由 ErrorCode 集中推导。
       唯一依据是开发文档 §9 的失败分类处置表——
       "重试还是降级"是错误分类的属性，不是某个工具的自由裁量。
       否则 6 个工具各自填 retryable，口径必然漂移。
    2. LLM 看到的永远是同一种 JSON 结构，编排层可以用纯代码
       按 code 分流（重试 / 降级 / 终止），不需要让模型去"理解"异常堆栈。

       补充 ：
    retryable指在 Agent 运行的宏观生命周期（Agent Loop）中，
    当前这个工具调用失败后，系统是否有价值、有资格带着“修正动作”再次发起相同或修正后的请求。

"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class ErrorCode(str, Enum):
    """错误分类。每一类对应 §9 中的一种处置策略。

    继承 str 是为了序列化后仍是可读字符串（如 "INVALID_PARAM"），
    方便直接写进 Trace JSONL 和 LLM 上下文。
    """

    INVALID_PARAM = "INVALID_PARAM"            # 参数非法：修参数或终止，不自动重试
    TIMEOUT = "TIMEOUT"                        # 超时：可有限重试（可调大 time budget）
    SOLVER_INFEASIBLE = "SOLVER_INFEASIBLE"    # 求解器不可行：结构性问题，原参数重试无意义
    DATA_MISSING = "DATA_MISSING"              # 数据/缓存缺失：检查数据源，严禁伪造结果
    LLM_INVALID_OUTPUT = "LLM_INVALID_OUTPUT"  # LLM 结构化输出非法：修复后重试一次
    INTERNAL = "INTERNAL"                      # 未预期内部错误：兜底，不自动重试


# §9 失败分类处置表的代码实现：retryable 的唯一事实来源。
# 新增 ErrorCode 时必须在这里登记，否则 KeyError 会立刻暴露遗漏（fail-fast）。
_RETRYABLE: dict[ErrorCode, bool] = {
    ErrorCode.INVALID_PARAM: False,      # 重复非法请求无价值
    ErrorCode.TIMEOUT: True,             # 超时不代表参数错误
    ErrorCode.SOLVER_INFEASIBLE: False,  # 结构不可行，应降级而非重试
    ErrorCode.DATA_MISSING: True,        # 数据可能稍后就绪（如缓存重建中）
    ErrorCode.LLM_INVALID_OUTPUT: True,  # 修复 prompt/输出后可再试一次
    ErrorCode.INTERNAL: False,           # 未知错误不盲目重试，先记录
}


class ToolError(BaseModel):
    """结构化错误。retryable 由 code 自动推导，禁止手工指定矛盾值。"""

    model_config = ConfigDict(frozen=True)  # 错误对象创建后不可变，防止传递过程中被改写

    code: ErrorCode
    message: str          # 给人和 LLM 看的定位信息：哪个参数、什么值、为什么不行
    retryable: bool

    @model_validator(mode="before")
    @classmethod
    def _fill_retryable(cls, values):
        """未显式给 retryable 时，按 _RETRYABLE 表自动推导。"""
        if isinstance(values, dict) and "retryable" not in values:
            code = values.get("code")
            code = ErrorCode(code)  # 允许传入字符串，统一转枚举
            values = {**values, "code": code, "retryable": _RETRYABLE[code]}
        return values

    @model_validator(mode="after")
    def _check_retryable_consistency(self):
        """显式给了 retryable 但与 §9 分类表矛盾 → 直接拒绝。

        这保证"retryable 由分类决定"这条纪律在运行时被强制，
        而不是只靠开发者自觉。
        """
        expected = _RETRYABLE[self.code]
        if self.retryable != expected:
            raise ValueError(
                f"retryable={self.retryable} 与错误分类 {self.code.value} 的处置策略"
                f"（应为 {expected}）矛盾；retryable 不允许工具自行改写"
            )
        return self


class ToolResult(BaseModel):
    """所有 Tool 的统一返回类型。

    状态一致性约束（after 校验器强制）：
        ok=True  ⇒ 不得携带 error
        ok=False ⇒ 必须携带 error（不允许"失败但没说为什么"）
    data 在失败时也允许存在（可携带部分结果/诊断信息），但编排层
    只有在 ok=True 时才应消费 data。
    """

    ok: bool
    data: dict | None = None
    error: ToolError | None = None

    "强校验器 创建toolresult对象后必定触发"
    @model_validator(mode="after")
    def _check_state_consistency(self):
        if self.ok and self.error is not None:
            raise ValueError("ok=True 时不得携带 error")
        if not self.ok and self.error is None:
            raise ValueError("ok=False 时必须携带 error")
        return self

    @classmethod
    def success(cls, data: dict) -> "ToolResult":
        """成功路径的唯一构造入口。"""
        return cls(ok=True, data=data, error=None)

    @classmethod
    def fail(cls, code: ErrorCode, message: str, data: dict | None = None) -> "ToolResult":
        """失败路径的唯一构造入口；retryable 由 code 自动推导。"""
        return cls(ok=False, data=data, error=ToolError(code=code, message=message))
