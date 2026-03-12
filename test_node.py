"""测试 GoalDeterminationNode - 使用真实大模型"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from nodes.goal_determination_node import (
    GoalDeterminationNode, 
    BrandProfile,
    GoalEnum
)

from openai import OpenAI
import json


class InternLLM:
    """包装 OpenAI 客户端为 LangChain 兼容接口"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "intern-latest"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    def invoke(self, prompt_value) -> str:
        """调用大模型"""
        # 处理不同类型的输入
        if hasattr(prompt_value, 'to_messages'):
            # ChatPromptValue 对象
            messages = prompt_value.to_messages()
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = "system" if msg.type == "system" else "user" if msg.type == "human" else "assistant"
                    formatted_messages.append({"role": role, "content": msg.content})
                else:
                    formatted_messages.append({"role": "user", "content": str(msg)})
        elif isinstance(prompt_value, dict):
            # 字典输入
            formatted_messages = []
            if "system" in prompt_value:
                formatted_messages.append({"role": "system", "content": prompt_value["system"]})
            content = prompt_value.get("user_instruction", "") or prompt_value.get("text", "")
            formatted_messages.append({"role": "user", "content": content})
        else:
            # 字符串输入
            formatted_messages = [{"role": "user", "content": str(prompt_value)}]
        
        # 调用 API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=0,
            stream=False
        )
        
        return response.choices[0].message.content
    
    def __call__(self, prompt_value) -> str:
        return self.invoke(prompt_value)


def test_with_real_llm():
    """使用真实大模型测试"""
    print("=" * 60)
    print("使用真实 LLM (intern-latest) 测试")
    print("=" * 60)
    
    # 配置 API
    API_KEY = "sk-Hv9eGdri29KNnv8WsOHxrcilcOARX8G81wP5ydnFix4IOdqx"  # 请替换为真实 token
    BASE_URL = "https://chat.intern-ai.org.cn/api/v1/"
    
    llm = InternLLM(api_key=API_KEY, base_url=BASE_URL)
    node = GoalDeterminationNode(llm)
    
    brand = BrandProfile(
        brand_name="新茶饮品牌",
        core_appeal="提升知名度",
        is_new_brand=True
    )
    
    # 测试1: 信息完整的指令
    print("\n测试1: 信息完整的指令")
    print("-" * 40)
    result = node.first_pass("中秋节做一场拉新活动，针对年轻人，预算充足")
    print(f"提取槽位:")
    print(f"  time_point: {result.slots.time_point}")
    print(f"  primary_goal: {result.slots.primary_goal}")
    print(f"  persona_hint: {result.slots.persona_hint}")
    print(f"  budget_level: {result.slots.budget_level}")
    print(f"  confidence: {result.slots.confidence}")
    print(f"缺失字段: {result.missing_fields}")
    print(f"需要澄清: {result.needs_clarification}")
    
    # 测试2: 信息不完整的指令
    print("\n测试2: 信息不完整的指令")
    print("-" * 40)
    result2 = node.first_pass("帮我策划一个活动")
    print(f"提取槽位:")
    print(f"  primary_goal: {result2.slots.primary_goal}")
    print(f"  time_point: {result2.slots.time_point}")
    print(f"  confidence: {result2.slots.confidence}")
    print(f"缺失字段: {result2.missing_fields}")
    
    if result2.needs_clarification:
        print("\n生成推荐选项...")
        recommendations = node.generate_recommendations("帮我策划一个活动", result2)
        for rec in recommendations.recommendations:
            print(f"\n  [{rec.field}] {rec.question}")
            for opt in rec.options[:3]:  # 只显示前3个
                print(f"    - {opt}")
    
    # 测试3: 规则引擎回退
    print("\n测试3: 规则引擎回退")
    print("-" * 40)
    result3 = node.run_interactive("中秋节做一场线下门店打卡活动", brand)
    print(f"输入: 中秋节做一场线下门店打卡活动")
    print(f"判定结果:")
    print(f"  primary_goal: {result3.primary_goal.value if result3.primary_goal else None}")
    print(f"  source: {result3.source}")
    print(f"  applied_rules: {result3.applied_rules}")


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("GoalDeterminationNode 真实 LLM 测试")
    print("*" * 60 + "\n")
    
    try:
        test_with_real_llm()
        print("\n" + "*" * 60)
        print("测试完成！")
        print("*" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
