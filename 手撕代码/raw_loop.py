import json
from typing import Callable, Dict, List, Any
from openai import OpenAI


# 1. 模拟定义一个本地业务工具
def get_traffic_speed(region_id: int) -> str:
    """获取指定区域的实时交通车速"""
    # 模拟真实业务逻辑
    if region_id == 3:
        return json.dumps({"region_id": 3, "speed_kmh": 25.4, "status": "congested"})
    return json.dumps({"region_id": region_id, "speed_kmh": 65.0, "status": "smooth"})


# 注册中心：将工具名称映射到真实的 Python 函数
TOOL_REGISTRY: Dict[str, Callable] = {
    "get_traffic_speed": get_traffic_speed
}


# 2. 核心：手写原生 Agent 循环 (Raw Loop)
def run_agent_raw_loop(user_goal: str, max_steps: int = 5) -> str:
    client = OpenAI()  # 假设环境变量已配置 API Key

    # 初始化上下文消息列表
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是一个智能交通助手，可以调用工具查询实时路况。"},
        {"role": "user", "content": user_goal}
    ]

    # 声明大模型能调用的工具 Schema 描述
    tools_definition = [{
        "type": "function",
        "function": {
            "name": "get_traffic_speed",
            "description": "获取指定区域的实时交通车速",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "integer", "description": "区域ID"}
                },
                "required": ["region_id"]
            }
        }
    }]

    # 进入主控制循环
    for step in range(max_steps):
        print(f"\n[Raw Loop 运行中] ── 第 {step + 1} 轮迭代 ──")

        try:
            # 1. 调用大模型
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools_definition,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # 2. 把大模型的响应追加到历史上下文中
            # 注意：在 OpenAI SDK 中，必须把原始 message 对象转成 dict 或直接存入
            messages.append(response_message)

            # 3. 检查模型是否决定调用工具
            tool_calls = response_message.tool_calls
            if not tool_calls:
                # 没有工具调用，说明大模型已经直接给出了最终文本回答
                print("🏁 Agent 任务完成，输出最终结论。")
                return response_message.content   #最后返回的就是这个 content

            # 4. 如果有工具调用，遍历并执行
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id

                print(f"🔧 [动作拦截] 模型请求调用工具: {func_name}, 参数: {func_args}")

                # 检查工具是否存在
                if func_name not in TOOL_REGISTRY:
                    tool_result = f"Error: Tool {func_name} not found."
                else:
                    try:
                        # 真正的本地函数执行
                        tool_result = TOOL_REGISTRY[func_name](**func_args)
                    except Exception as e:
                        # 异常捕获：将错误作为观测结果返回，让大模型自我纠错
                        tool_result = f"Tool Execution Error: {str(e)}"

                print(f"📦 [观测反馈] 工具执行结果: {tool_result}")

                # 5. 将工具执行结果以 tool 角色写回上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(tool_result)
                })

        except Exception as e:
            print(f"❌ [系统异常] 循环内部发生错误: {e}")
            break

    return "Error: Reached maximum steps without finishing the task."


# ── 测试运行 ──
if __name__ == "__main__":
    final_answer = run_agent_raw_loop("帮我查一下 3 号区域的实时交通车速情况怎么样？")
    print(f"\n最终结果:\n{final_answer}")