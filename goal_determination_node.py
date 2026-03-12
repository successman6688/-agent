"""
工作流节点1：确认唯一首要目标 (Goal Determination Node)

交互式两轮提取流程：
1. 第一轮提取所有槽位
2. 检查缺失字段，生成推荐询问用户
3. 用户补充后，第二轮提取完整槽位
4. 规则引擎判定目标

增强功能：记忆存储 + 审计日志
"""

from typing import Optional, List, Literal, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ 数据模型 ============

class GoalEnum(str, Enum):
    """首要目标枚举"""
    ACQUIRE_USER = "acquire_user"           # 拉新
    ACTIVATE_DORMANT = "activate_dormant"   # 促活
    UGC_CONTENT = "ugc_content"             # 内容产出
    BRAND_AWARENESS = "brand_awareness"     # 品宣
    CONVERT_PURCHASE = "convert_purchase"   # 转化


class AssetTypeEnum(str, Enum):
    """信任资产类型枚举"""
    CONTENT = "content"         # 内容资产
    RELATIONSHIP = "relationship"  # 关系资产
    DATA = "data"               # 数据资产


class BudgetLevelEnum(str, Enum):
    """预算等级枚举"""
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class ExtractedSlots(BaseModel):
    """LLM提取的结构化槽位"""
    model_config = {"use_enum_values": True}
    
    # 品牌信息（若用户在指令中提及）
    brand_name: Optional[str] = Field(None, description="品牌名称（如用户提及）")
    core_appeal_hint: Optional[str] = Field(None, description="品牌核心诉求提示（如用户提及）")
    
    # 活动信息
    time_point: Optional[str] = Field(None, description="活动时间节点")
    primary_goal: Optional[GoalEnum] = Field(None, description="首要目标")
    asset_type: Optional[List[AssetTypeEnum]] = Field(None, description="信任资产类型")
    scene_hint: Optional[str] = Field(None, description="场景关键词")
    persona_hint: Optional[str] = Field(None, description="人群关键词")
    budget_level: Optional[BudgetLevelEnum] = Field(None, description="预算等级")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="提取置信度")
    
    def get_missing_fields(self, core_only: bool = True) -> List[str]:
        """
        获取为null的关键字段列表
        
        Args:
            core_only: 是否只检查核心字段（primary_goal, time_point）
                      若为False，则检查所有字段
        """
        missing = []
        if self.primary_goal is None:
            missing.append("primary_goal")
        if self.time_point is None:
            missing.append("time_point")
        
        if not core_only:
            if self.scene_hint is None:
                missing.append("scene_hint")
            if self.persona_hint is None:
                missing.append("persona_hint")
            if self.budget_level is None:
                missing.append("budget_level")
            if self.brand_name is None:
                missing.append("brand_name")
        
        return missing


class RecommendationItem(BaseModel):
    """单个推荐项"""
    field: str = Field(description="字段名")
    question: str = Field(description="询问用户的问题")
    options: List[str] = Field(description="推荐选项")


class RecommendationResult(BaseModel):
    """推荐结果"""
    recommendations: List[RecommendationItem] = Field(description="推荐列表")
    original_query: str = Field(description="原始查询")


class BrandProfile(BaseModel):
    """品牌信息"""
    brand_name: str
    core_appeal: str  # 拉新破圈/提升复购/提升知名度/内容传播/促进转化
    is_new_brand: bool = False
    is_new_product: bool = False


class FirstPassResult(BaseModel):
    """第一轮提取结果"""
    slots: ExtractedSlots
    missing_fields: List[str]
    needs_clarification: bool


class GoalDeterminationState(BaseModel):
    """节点输出状态"""
    model_config = {"use_enum_values": True}
    
    primary_goal: GoalEnum = Field(description="最终确定的首要目标")
    source: Literal["llm_direct", "rule_engine", "human_override"] = Field(description="判定来源")
    confidence: float = Field(description="置信度")
    extracted_slots: ExtractedSlots = Field(description="提取的槽位")
    applied_rules: Optional[List[str]] = Field(None, description="应用的规则")
    session_id: Optional[str] = Field(None, description="会话ID")


class ExtractionMemory(BaseModel):
    """提取记忆记录"""
    session_id: str = Field(description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="记录时间")
    user_instruction: str = Field(description="用户原始指令")
    first_pass_slots: ExtractedSlots = Field(description="第一轮提取结果")
    missing_fields: List[str] = Field(description="缺失字段")
    user_supplements: Optional[Dict[str, Any]] = Field(None, description="用户补充信息")
    final_slots: Optional[ExtractedSlots] = Field(None, description="最终槽位")
    final_goal: Optional[GoalEnum] = Field(None, description="最终判定目标")
    source: Optional[str] = Field(None, description="判定来源")
    
    def to_dict(self) -> dict:
        """转换为字典用于存储"""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_instruction": self.user_instruction,
            "first_pass_slots": self.first_pass_slots.model_dump(),
            "missing_fields": self.missing_fields,
            "user_supplements": self.user_supplements,
            "final_slots": self.final_slots.model_dump() if self.final_slots else None,
            "final_goal": self.final_goal,
            "source": self.source
        }


# ============ 提示词模板 ============

# ============ 提示词模板 ============

SLOT_EXTRACTION_PROMPT = """你是 StrategyAI 的意图解析器。从用户指令中提取结构化槽位。

需提取的槽位（只从用户指令中提取，不要猜测）：
- brand_name: 品牌名称（如用户提及"我们是XX品牌"或"帮XX品牌策划"，无法判断为null）
- core_appeal_hint: 品牌核心诉求提示（如用户提及"我们主要想拉新""目标是提升知名度"，无法判断为null）
- time_point: 活动时间节点（如 '中秋节'/'下个月会员日'，无法判断为null）
- primary_goal: 首要目标（枚举：acquire_user / activate_dormant / ugc_content / brand_awareness / convert_purchase，无法判断为null）
- asset_type: 信任资产类型（数组：内容资产/口碑内容 → content，关系资产/社群/KOL → relationship，数据资产/会员数据 → data，无法判断则为 null）
- scene_hint: 场景关键词（如用户提及"线下门店""社交媒体"等，无法判断为null）
- persona_hint: 人群关键词（如用户提及"年轻人""会员"等，无法判断为null）  
- budget_level: 预算等级（low / mid / high，无法判断则为 null）

枚举值映射：
- primary_goal: acquire_user(拉新)/activate_dormant(促活)/ugc_content(内容)/brand_awareness(品宣)/convert_purchase(转化)
- asset_type: content(内容资产)/relationship(关系资产)/data(数据资产)
- budget_level: low(低预算/省钱)/mid(中等预算)/high(高预算/充足)

重要规则：
1. 只从用户指令中提取信息，**不要**使用外部知识
2. 如果用户指令中**明确提到**目标（如"拉新""促活""转化"），提取对应枚举值
3. 如果**未明确提到**目标，primary_goal 返回 null
4. 品牌信息仅在用户明确提及时提取，否则为 null
5. confidence 表示你对提取结果的信心（0.0-1.0）

用户指令：{user_instruction}

{format_instructions}
"""

RECOMMENDATION_PROMPT = """你是 StrategyAI 的推荐助手。基于用户指令和缺失的字段，生成智能推荐选项。

用户原始指令：{user_instruction}

第一轮提取结果：
{extracted_slots}

缺失字段：{missing_fields}

请为每个缺失字段生成：
1. 一个友好的询问问题
2. 3-5个推荐选项（基于上下文推测用户可能的需求）

输出格式：
{format_instructions}
"""

SECOND_PASS_PROMPT = """你是 StrategyAI 的意图解析器。基于用户原始指令和补充信息，重新提取完整槽位。

用户原始指令：{user_instruction}

用户补充信息：
{user_supplements}

请重新提取所有槽位，优先使用用户补充的信息。

需提取的槽位：
- brand_name: 品牌名称
- core_appeal_hint: 品牌核心诉求提示
- time_point: 活动时间节点
- primary_goal: 首要目标（枚举：acquire_user / activate_dormant / ugc_content / brand_awareness / convert_purchase）
- asset_type: 信任资产类型（数组：content / relationship / data）
- scene_hint: 场景关键词
- persona_hint: 人群关键词
- budget_level: 预算等级（low / mid / high）
- confidence: 提取置信度（0.0-1.0）

{format_instructions}
"""


# ============ 节点实现 ============

class GoalDeterminationNode:
    """
    工作流节点1：确认唯一首要目标（交互式两轮提取 + 记忆日志）
    
    流程：
    1. 第一轮提取所有槽位
    2. 检查缺失字段，生成推荐询问用户
    3. 用户补充后，第二轮提取完整槽位
    4. 规则引擎判定目标
    5. 记录记忆和审计日志
    """
    
    def __init__(self, llm, memory_store: Optional[List] = None):
        """
        Args:
            llm: LangChain LLM instance 或兼容接口
            memory_store: 可选的外部记忆存储列表
        """
        self.llm = llm
        self.memory_store = memory_store or []
        self.session_counter = 0
        
        # 第一轮提取链
        self.first_pass_parser = PydanticOutputParser(pydantic_object=ExtractedSlots)
        self.first_pass_prompt = ChatPromptTemplate.from_template(
            SLOT_EXTRACTION_PROMPT,
            partial_variables={"format_instructions": self.first_pass_parser.get_format_instructions()}
        )
        self.first_pass_chain = self.first_pass_prompt | self.llm | self.first_pass_parser
        
        # 推荐生成链
        self.recommendation_parser = PydanticOutputParser(pydantic_object=RecommendationResult)
        self.recommendation_prompt = ChatPromptTemplate.from_template(
            RECOMMENDATION_PROMPT,
            partial_variables={"format_instructions": self.recommendation_parser.get_format_instructions()}
        )
        self.recommendation_chain = self.recommendation_prompt | self.llm | self.recommendation_parser
        
        # 第二轮提取链
        self.second_pass_prompt = ChatPromptTemplate.from_template(
            SECOND_PASS_PROMPT,
            partial_variables={"format_instructions": self.first_pass_parser.get_format_instructions()}
        )
        self.second_pass_chain = self.second_pass_prompt | self.llm | self.first_pass_parser
        
        logger.info("GoalDeterminationNode initialized")
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        self.session_counter += 1
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.session_counter}"
    
    def _save_memory(self, memory: ExtractionMemory):
        """保存记忆"""
        self.memory_store.append(memory)
        logger.info(f"[Memory Saved] Session: {memory.session_id}, Goal: {memory.final_goal}")
    
    def _log_extraction(self, session_id: str, phase: str, data: dict):
        """记录提取日志"""
        logger.info(f"[Extraction] Session: {session_id}, Phase: {phase}, Data: {json.dumps(data, ensure_ascii=False, default=str)}")
    
    def first_pass(self, user_instruction: str, session_id: Optional[str] = None) -> FirstPassResult:
        """
        第一轮提取：从用户指令提取所有槽位
        
        Args:
            user_instruction: 用户原始指令
            session_id: 可选的会话ID
            
        Returns:
            FirstPassResult: 包含提取的槽位和缺失字段信息
        """
        sid = session_id or self._generate_session_id()
        logger.info(f"[First Pass Start] Session: {sid}, Query: {user_instruction}")
        
        slots = self.first_pass_chain.invoke({
            "user_instruction": user_instruction
        })
        
        missing_fields = slots.get_missing_fields()
        
        self._log_extraction(sid, "first_pass", {
            "slots": slots.model_dump(),
            "missing_fields": missing_fields
        })
        
        return FirstPassResult(
            slots=slots,
            missing_fields=missing_fields,
            needs_clarification=len(missing_fields) > 0
        )
    
    def generate_recommendations(
        self, 
        user_instruction: str, 
        first_pass_result: FirstPassResult
    ) -> RecommendationResult:
        """
        为缺失字段生成推荐选项
        
        Args:
            user_instruction: 用户原始指令
            first_pass_result: 第一轮提取结果
            
        Returns:
            RecommendationResult: 推荐列表
        """
        if not first_pass_result.needs_clarification:
            return RecommendationResult(
                recommendations=[],
                original_query=user_instruction
            )
        
        return self.recommendation_chain.invoke({
            "user_instruction": user_instruction,
            "extracted_slots": first_pass_result.slots.model_dump_json(),
            "missing_fields": ", ".join(first_pass_result.missing_fields)
        })
    
    def second_pass(
        self, 
        user_instruction: str, 
        user_supplements: Dict[str, Any]
    ) -> ExtractedSlots:
        """
        第二轮提取：结合用户补充信息重新提取
        
        Args:
            user_instruction: 用户原始指令
            user_supplements: 用户补充的字段值
            
        Returns:
            ExtractedSlots: 完整的槽位信息
        """
        supplements_text = "\n".join([
            f"- {field}: {value}" 
            for field, value in user_supplements.items()
        ])
        
        return self.second_pass_chain.invoke({
            "user_instruction": user_instruction,
            "user_supplements": supplements_text
        })
    
    def determine_goal(
        self, 
        slots: ExtractedSlots, 
        brand_profile: BrandProfile
    ) -> GoalDeterminationState:
        """
        判定最终目标
        
        Args:
            slots: 提取的槽位
            brand_profile: 品牌信息
            
        Returns:
            GoalDeterminationState: 最终目标判定结果
        """
        # 若primary_goal清晰，直接采用
        if slots.primary_goal is not None and slots.confidence >= 0.8:
            return GoalDeterminationState(
                primary_goal=slots.primary_goal,
                source="llm_direct",
                confidence=slots.confidence,
                extracted_slots=slots
            )
        
        # 规则引擎判定
        goal, rules = self._rule_based_resolve(slots, brand_profile)
        
        return GoalDeterminationState(
            primary_goal=goal,
            source="rule_engine",
            confidence=0.7 if slots.confidence else 0.6,
            extracted_slots=slots,
            applied_rules=rules
        )
    
    def run_interactive(
        self, 
        user_instruction: str, 
        brand_profile: BrandProfile,
        user_supplements: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> GoalDeterminationState:
        """
        交互式完整流程（供外部调用）
        
        Args:
            user_instruction: 用户原始指令
            brand_profile: 品牌信息
            user_supplements: 用户补充的字段值（如有）
            session_id: 可选的会话ID
            
        Returns:
            GoalDeterminationState: 最终目标判定结果
        """
        sid = session_id or self._generate_session_id()
        logger.info(f"[Interactive Start] Session: {sid}")
        
        # 初始化记忆
        memory = ExtractionMemory(
            session_id=sid,
            user_instruction=user_instruction,
            first_pass_slots=ExtractedSlots(),  # 临时占位
            missing_fields=[]
        )
        
        if user_supplements:
            # 有补充信息，直接第二轮提取
            logger.info(f"[Second Pass] Session: {sid}, Supplements: {user_supplements}")
            slots = self.second_pass(user_instruction, user_supplements)
            memory.user_supplements = user_supplements
        else:
            # 第一轮提取
            first_result = self.first_pass(user_instruction, sid)
            slots = first_result.slots
            memory.first_pass_slots = slots
            memory.missing_fields = first_result.missing_fields
        
        # 判定目标
        result = self.determine_goal(slots, brand_profile)
        result.session_id = sid
        
        # 更新记忆
        memory.final_slots = slots
        memory.final_goal = result.primary_goal
        memory.source = result.source
        self._save_memory(memory)
        
        logger.info(f"[Interactive Complete] Session: {sid}, Goal: {result.primary_goal}, Source: {result.source}")
        
        return result
    
    def _rule_based_resolve(
        self, 
        slots: ExtractedSlots, 
        brand: BrandProfile
    ) -> tuple[GoalEnum, List[str]]:
        """
        基于规则的自动判定逻辑
        
        规则优先级（从高到低）：
        1. 场景关键词匹配
        2. 人群关键词 + 品牌诉求匹配  
        3. 品牌核心诉求匹配
        4. 默认回退
        """
        scene = slots.scene_hint or ""
        persona = slots.persona_hint or ""
        
        # Rule 1: 场景匹配
        if any(kw in scene for kw in ["促销", "下单", "会员转化", "大促", "秒杀"]):
            return GoalEnum.CONVERT_PURCHASE, ["RULE-001:场景含促销关键词"]
        
        if any(kw in scene for kw in ["打卡", "晒单", "种草", "UGC", "分享"]):
            return GoalEnum.UGC_CONTENT, ["RULE-002:场景含内容传播关键词"]
        
        # Rule 2: 人群 + 品牌诉求匹配
        if "休眠" in persona or "沉睡" in persona or "老用户" in persona:
            if "复购" in brand.core_appeal:
                return GoalEnum.ACTIVATE_DORMANT, ["RULE-003:休眠用户+复购诉求"]
        
        # Rule 3: 品牌核心诉求匹配
        appeal_map = {
            "拉新破圈": GoalEnum.ACQUIRE_USER,
            "提升复购": GoalEnum.ACTIVATE_DORMANT,
            "提升知名度": GoalEnum.BRAND_AWARENESS,
            "内容传播": GoalEnum.UGC_CONTENT,
            "促进转化": GoalEnum.CONVERT_PURCHASE,
        }
        if brand.core_appeal in appeal_map:
            goal = appeal_map[brand.core_appeal]
            return goal, [f"RULE-004:品牌核心诉求匹配-{brand.core_appeal}"]
        
        # Rule 4: 新品牌/新品默认品宣
        if brand.is_new_brand or brand.is_new_product:
            return GoalEnum.BRAND_AWARENESS, ["RULE-005:新品牌/新品默认品宣"]
        
        # Fallback
        return GoalEnum.BRAND_AWARENESS, ["RULE-FALLBACK:默认品宣"]


# ============ 使用示例 ============

if __name__ == "__main__":
    # 示例：使用 ChatOpenAI
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    node = GoalDeterminationNode(llm)
    
    brand = BrandProfile(
        brand_name="新茶饮品牌",
        core_appeal="提升知名度",
        is_new_brand=True
    )
    
    # 示例1: 信息完整的指令
    print("=" * 60)
    print("示例1: 信息完整的指令")
    result = node.run_interactive("中秋节做一场拉新活动，针对年轻人，预算充足", brand)
    print(f"目标: {result.primary_goal.value}, 来源: {result.source}")
    print(f"提取槽位: {result.extracted_slots.model_dump_json(indent=2)}")
    
    # 示例2: 信息不完整的指令 - 第一轮提取
    print("\n" + "=" * 60)
    print("示例2: 信息不完整的指令 - 第一轮提取")
    first_result = node.first_pass("帮我策划一个活动")
    print(f"第一轮提取: {first_result.slots.model_dump_json(indent=2)}")
    print(f"缺失字段: {first_result.missing_fields}")
    
    if first_result.needs_clarification:
        # 生成推荐
        recommendations = node.generate_recommendations("帮我策划一个活动", first_result)
        print(f"\n推荐选项:")
        for rec in recommendations.recommendations:
            print(f"  [{rec.field}] {rec.question}")
            for opt in rec.options:
                print(f"    - {opt}")
        
        # 模拟用户补充
        user_supplements = {
            "primary_goal": "brand_awareness",
            "time_point": "中秋节",
            "scene_hint": "线下门店",
            "budget_level": "mid"
        }
        
        # 第二轮提取
        print(f"\n用户补充: {user_supplements}")
        final_result = node.run_interactive("帮我策划一个活动", brand, user_supplements)
        print(f"最终目标: {final_result.primary_goal.value}")
        print(f"完整槽位: {final_result.extracted_slots.model_dump_json(indent=2)}")
