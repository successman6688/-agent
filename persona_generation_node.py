"""
工作流节点2：用户画像生成 (Persona Generation Node)

功能：
1. 向量检索品牌历史相似活动（Top5）
2. 对每个活动查询真实用户数据
3. 数据分析生成统计值（人口统计、行为偏好、痛点爽点）
4. 输出5个活动及其对应的用户画像卡片

数据驱动：使用 SQL/Python 从真实数据提取统计值
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

# 导入 MCP 工具
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.data_analyzer import (
    DataAnalyzer, 
    DemographicStats, 
    BehaviorStats, 
    PainPleasureAnalysis
)


# ============ 数据模型 ============

class PersonaCard(BaseModel):
    """用户画像卡片"""
    model_config = {"use_enum_values": True}
    
    persona_name: str = Field(description="人群名称（如：Z世代新锐白领）")
    demographics: Dict[str, Any] = Field(description="人口统计特征（年龄、性别、收入等）")
    core_traits: List[str] = Field(description="核心特征标签")
    pain_points: List[str] = Field(description="核心痛点")
    pleasure_points: List[str] = Field(description="爽点/需求点")
    behavior_prefs: Dict[str, Any] = Field(description="行为偏好（渠道、内容、时间等）")
    consumption_insights: str = Field(description="消费洞察")


class CoreScenario(BaseModel):
    """核心场景"""
    scenario_name: str = Field(description="场景名称")
    scenario_desc: str = Field(description="场景描述")
    trigger_moment: str = Field(description="触发时刻")
    user_motivation: str = Field(description="用户动机")
    expected_outcome: str = Field(description="期望结果")


class BrandKnowledgeItem(BaseModel):
    """品牌知识库条目"""
    category: str = Field(description="知识类别（场景/人群/产品）")
    content: str = Field(description="知识内容")
    relevance_score: float = Field(default=0.0, description="相关度分数")
    source: Optional[str] = Field(None, description="来源")


class HistoricalCampaign(BaseModel):
    """历史活动数据"""
    campaign_name: str = Field(description="活动名称")
    target_goal: str = Field(description="活动目标")
    target_persona: Optional[str] = Field(None, description="目标人群")
    performance: Dict[str, Any] = Field(default_factory=dict, description="活动效果数据")
    insights: Optional[str] = Field(None, description="经验洞察")


class CampaignWithPersona(BaseModel):
    """活动及其对应的用户画像"""
    campaign: HistoricalCampaign = Field(description="历史活动信息")
    persona_card: PersonaCard = Field(description="该活动的用户画像卡片")
    data_stats: Dict[str, Any] = Field(description="数据统计详情")


class PersonaGenerationState(BaseModel):
    """节点输出状态 - 包含5个活动及其画像"""
    model_config = {"use_enum_values": True}
    
    campaigns_with_personas: List[CampaignWithPersona] = Field(description="5个活动及其画像")
    scene_hint: Optional[str] = Field(None, description="场景关键词")
    primary_goal: Optional[str] = Field(None, description="首要目标")
    session_id: Optional[str] = Field(None, description="会话ID")


class PersonaGenerationMemory(BaseModel):
    """画像生成记忆记录"""
    session_id: str = Field(description="会话ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="记录时间")
    brand_name: str = Field(description="品牌名称")
    primary_goal: str = Field(description="首要目标")
    scene_hint: Optional[str] = Field(None, description="场景提示")
    persona_hint: Optional[str] = Field(None, description="人群提示")
    generated_persona: PersonaCard = Field(description="生成的画像")
    core_scenario: CoreScenario = Field(description="核心场景")
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "brand_name": self.brand_name,
            "primary_goal": self.primary_goal,
            "scene_hint": self.scene_hint,
            "persona_hint": self.persona_hint,
            "generated_persona": self.generated_persona.model_dump(),
            "core_scenario": self.core_scenario.model_dump()
        }


# ============ 提示词模板 ============

KNOWLEDGE_RETRIEVAL_PROMPT = """你是 StrategyAI 的品牌知识检索专家。

品牌信息：
- 品牌名称：{brand_name}
- 核心诉求：{core_appeal}

活动信息：
- 首要目标：{primary_goal}
- 场景提示：{scene_hint}
- 人群提示：{persona_hint}

请从品牌知识库中检索以下内容：
1. 与场景提示最匹配的品牌使用场景（3-5个）
2. 与目标最匹配的目标人群类型（2-3个）
3. 相关的产品/服务卖点

输出格式：
{format_instructions}
"""

PERSONA_GENERATION_PROMPT = """你是 StrategyAI 的用户画像专家。

基于以下信息生成目标人群画像卡片：

品牌信息：
- 品牌名称：{brand_name}
- 核心诉求：{core_appeal}

活动信息：
- 首要目标：{primary_goal}
- 场景提示：{scene_hint}
- 人群提示：{persona_hint}

检索到的品牌知识：
{retrieved_knowledge}

相似历史活动：
{similar_campaigns}

请生成：
1. 用户画像卡片（人群名称、特征、痛点、爽点、行为偏好）
2. 核心场景（场景名称、描述、触发时刻、动机、期望结果）

输出格式：
{format_instructions}
"""


# ============ 节点实现 ============

class PersonaGenerationNode:
    """
    工作流节点2：用户画像生成（数据驱动版）
    
    流程：
    1. 向量检索品牌历史相似活动（Top5）
    2. 对每个活动：
       - 查询参与用户数据
       - 分析人口统计特征
       - 分析行为偏好
       - 提取痛点爽点
       - 生成画像卡片
    3. 输出5个活动及其画像
    """
    
    def __init__(self, llm=None, data_analyzer: Optional[DataAnalyzer] = None, memory_store: Optional[List] = None):
        """
        Args:
            llm: LLM实例（可选，用于增强描述）
            data_analyzer: 数据分析工具实例
            memory_store: 记忆存储列表
        """
        self.llm = llm
        self.data_analyzer = data_analyzer or DataAnalyzer()
        self.memory_store = memory_store or []
        self.session_counter = 0
        
        logger.info("PersonaGenerationNode initialized (data-driven)")
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        self.session_counter += 1
        return f"persona_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.session_counter}"
    
    def _save_memory(self, memory: PersonaGenerationMemory):
        """保存记忆"""
        self.memory_store.append(memory)
        logger.info(f"[Memory Saved] Session: {memory.session_id}, Persona: {memory.generated_persona.persona_name}")
    
    def search_similar_campaigns(
        self,
        brand_id: str,
        scene_hint: str,
        primary_goal: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        向量检索相似历史活动
        
        Args:
            brand_id: 品牌ID
            scene_hint: 场景关键词
            primary_goal: 首要目标
            top_k: 返回数量
            
        Returns:
            List[Dict]: 相似活动列表（包含相似度分数）
        """
        return self.data_analyzer.search_similar_campaigns(
            scene_hint=scene_hint,
            brand_id=brand_id,
            primary_goal=primary_goal,
            top_k=top_k
        )
    
    def generate_persona_from_data(
        self,
        campaign_data: Dict[str, Any]
    ) -> CampaignWithPersona:
        """
        基于真实数据生成活动对应的用户画像
        
        Args:
            campaign_data: 活动数据（包含campaign_id等）
            
        Returns:
            CampaignWithPersona: 活动及其画像
        """
        camp_id = campaign_data["campaign_id"]
        logger.info(f"[Generate Persona] Campaign: {camp_id}")
        
        # 1. 获取活动参与用户
        participants = self.data_analyzer.get_campaign_participants(camp_id)
        
        # 2. 分析人口统计
        demo_stats = self.data_analyzer.analyze_demographics(participants)
        
        # 3. 分析行为偏好
        behavior_stats = self.data_analyzer.analyze_behavior_patterns(participants)
        
        # 4. 提取痛点爽点
        pain_pleasure = self.data_analyzer.extract_pain_pleasure_points(camp_id)
        
        # 5. 生成特征标签
        traits = self.data_analyzer.generate_persona_traits(demo_stats, behavior_stats, pain_pleasure)
        
        # 6. 构建 HistoricalCampaign 对象
        campaign = HistoricalCampaign(
            campaign_name=campaign_data["campaign_name"],
            target_goal=campaign_data["target_goal"],
            target_persona=", ".join(traits[:3]),
            performance={
                "participants": campaign_data["participants_count"],
                "ugc_count": campaign_data["ugc_count"],
                "conversion_rate": campaign_data["conversion_rate"],
                "similarity_score": campaign_data["similarity_score"]
            },
            insights=f"相似度{campaign_data['similarity_score']:.2f}，主要人群特征：{', '.join(traits)}"
        )
        
        # 7. 构建 PersonaCard
        persona = PersonaCard(
            persona_name=self._generate_persona_name(traits, demo_stats),
            demographics={
                "age_distribution": demo_stats.age_distribution,
                "gender_ratio": demo_stats.gender_ratio,
                "income_distribution": demo_stats.income_distribution,
                "city_tier": demo_stats.city_tier_distribution
            },
            core_traits=traits,
            pain_points=[p["point"] for p in pain_pleasure.top_pain_points[:5]],
            pleasure_points=[p["point"] for p in pain_pleasure.top_pleasure_points[:5]],
            behavior_prefs={
                "channels": behavior_stats.channel_preferences,
                "content": behavior_stats.content_preferences,
                "active_time": behavior_stats.active_time_distribution,
                "avg_session_duration": f"{behavior_stats.avg_session_duration}分钟",
                "engagement_rate": f"{behavior_stats.engagement_rate:.1%}"
            },
            consumption_insights=self._generate_consumption_insights(demo_stats, behavior_stats)
        )
        
        # 8. 构建数据详情
        data_stats = {
            "participant_count": len(participants),
            "demographics": demo_stats.model_dump(),
            "behavior": behavior_stats.model_dump(),
            "pain_pleasure": pain_pleasure.model_dump()
        }
        
        return CampaignWithPersona(
            campaign=campaign,
            persona_card=persona,
            data_stats=data_stats
        )
    
    def _generate_persona_name(self, traits: List[str], demo: DemographicStats) -> str:
        """根据特征和人口统计生成人群名称"""
        # 基于主要年龄段
        age_max = max(demo.age_distribution.items(), key=lambda x: x[1])
        age_label = "Z世代" if age_max[0] in ["18-24岁", "25-30岁"] else "新锐白领" if age_max[0] in ["31-35岁"] else "品质人群"
        
        # 基于主要特征
        feature = traits[0] if traits else "探索者"
        
        return f"{age_label}{feature}"
    
    def _generate_consumption_insights(self, demo: DemographicStats, behavior: BehaviorStats) -> str:
        """生成消费洞察"""
        insights = []
        
        # 基于城市分布
        tier1 = demo.city_tier_distribution.get("一线城市", 0)
        if tier1 > 40:
            insights.append("一线城市集中，消费能力强")
        
        # 基于互动率
        if behavior.engagement_rate > 0.3:
            insights.append("高互动意愿，适合UGC策略")
        
        # 基于渠道偏好
        top_channel = max(behavior.channel_preferences.items(), key=lambda x: x[1])
        insights.append(f"主要活跃在{top_channel[0]}，占比{top_channel[1]:.1f}%")
        
        return "；".join(insights)
    
    def generate_persona(
        self,
        brand_profile: Any,  # BrandProfile
        primary_goal: str,
        scene_hint: Optional[str],
        persona_hint: Optional[str],
        retrieved_knowledge: List[BrandKnowledgeItem],
        similar_campaigns: List[HistoricalCampaign]
    ) -> PersonaGenerationState:
        """
        生成用户画像和核心场景
        
        实际实现中这里应该调用LLM
        这里使用模板生成演示
        """
        logger.info(f"[Persona Generation] Goal: {primary_goal}, Scene: {scene_hint}")
        
        # 根据目标确定画像方向
        if primary_goal == "acquire_user":
            persona = PersonaCard(
                persona_name="Z世代新锐探索者",
                demographics={"age": "18-28", "gender": "男女不限", "income": "中等偏上"},
                core_traits=["追求新鲜", "社交活跃", "注重颜值", "乐于分享"],
                pain_points=["选择困难", "担心踩雷", "渴望被认同"],
                pleasure_points=["独特体验", "社交货币", "身份认同"],
                behavior_prefs={
                    "channels": ["小红书", "抖音", "朋友圈"],
                    "content": ["高颜值", "真实测评", "限时福利"],
                    "time": ["周末", "午休", "晚间"]
                },
                consumption_insights="愿意为体验和颜值付费，重视朋友推荐"
            )
            scenario = CoreScenario(
                scenario_name="新品尝鲜打卡",
                scenario_desc="在门店体验新品，拍照分享到社交平台",
                trigger_moment="周末休闲时光，朋友聚会",
                user_motivation="尝试新鲜事物，获得社交认同",
                expected_outcome="获得优质内容和优惠券"
            )
        elif primary_goal == "activate_dormant":
            persona = PersonaCard(
                persona_name="沉睡老会员",
                demographics={"age": "25-40", "gender": "男女不限", "income": "中高收入"},
                core_traits=["价格敏感", "注重实惠", "品牌忠诚度高"],
                pain_points=["感觉被遗忘", "优惠力度不够", "缺乏新鲜感"],
                pleasure_points=["专属福利", "VIP待遇", "惊喜体验"],
                behavior_prefs={
                    "channels": ["微信", "短信", "APP推送"],
                    "content": ["限时折扣", "会员专属", "积分翻倍"],
                    "time": ["工作日晚间", "周末"]
                },
                consumption_insights="对专属福利敏感，需要被重视的感觉"
            )
            scenario = CoreScenario(
                scenario_name="会员专属唤醒",
                scenario_desc="收到专属福利通知，重新激活消费",
                trigger_moment="收到个性化推送",
                user_motivation="不想错过专属优惠",
                expected_outcome="获得实惠，重拾品牌好感"
            )
        else:
            # 默认画像
            persona = PersonaCard(
                persona_name="品质生活追求者",
                demographics={"age": "22-35", "gender": "男女不限", "income": "中等收入"},
                core_traits=["注重品质", "理性消费", "追求效率"],
                pain_points=["信息过载", "时间有限", "选择困难"],
                pleasure_points=["品质保证", "便捷体验", "性价比"],
                behavior_prefs={
                    "channels": ["微信", "小红书", "抖音"],
                    "content": ["真实评价", "专业测评", "使用教程"],
                    "time": ["午休", "晚间", "周末"]
                },
                consumption_insights="注重性价比，信任专业推荐"
            )
            scenario = CoreScenario(
                scenario_name="日常品质消费",
                scenario_desc="在日常生活中选择品牌产品",
                trigger_moment="有消费需求时",
                user_motivation="满足日常需求，追求品质",
                expected_outcome="获得满意的产品体验"
            )
        
        return PersonaGenerationState(
            persona_card=persona,
            core_scenario=scenario,
            retrieved_knowledge=retrieved_knowledge,
            similar_campaigns=similar_campaigns
        )
    
    def run(
        self,
        brand_profile: Any,  # BrandProfile
        primary_goal: str,
        scene_hint: Optional[str] = None,
        persona_hint: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> PersonaGenerationState:
        """
        执行节点流程 - 数据驱动生成5个活动及其画像
        
        Args:
            brand_profile: 品牌信息
            primary_goal: 首要目标
            scene_hint: 场景提示
            persona_hint: 人群提示
            session_id: 可选的会话ID
            
        Returns:
            PersonaGenerationState: 5个活动及其对应的用户画像
        """
        sid = session_id or self._generate_session_id()
        logger.info(f"[Persona Generation Start] Session: {sid}, Scene: {scene_hint}")
        
        # 1. 向量检索相似活动（Top5）
        similar_campaigns = self.search_similar_campaigns(
            brand_id=getattr(brand_profile, 'brand_id', 'BR001'),
            scene_hint=scene_hint or "",
            primary_goal=primary_goal,
            top_k=5
        )
        
        logger.info(f"[Retrieved] {len(similar_campaigns)} similar campaigns")
        
        # 2. 对每个活动生成数据驱动的画像
        campaigns_with_personas = []
        for camp_data in similar_campaigns:
            campaign_with_persona = self.generate_persona_from_data(camp_data)
            campaigns_with_personas.append(campaign_with_persona)
            logger.info(f"[Persona Generated] {camp_data['campaign_name']} -> {campaign_with_persona.persona_card.persona_name}")
        
        # 3. 构建结果
        result = PersonaGenerationState(
            campaigns_with_personas=campaigns_with_personas,
            scene_hint=scene_hint,
            primary_goal=primary_goal,
            session_id=sid
        )
        
        # 4. 保存记忆（只保存第一个作为代表）
        if campaigns_with_personas:
            first = campaigns_with_personas[0]
            memory = PersonaGenerationMemory(
                session_id=sid,
                brand_name=brand_profile.brand_name,
                primary_goal=primary_goal,
                scene_hint=scene_hint,
                persona_hint=persona_hint,
                generated_persona=first.persona_card,
                core_scenario=CoreScenario(
                    scenario_name=first.campaign.campaign_name,
                    scenario_desc=f"相似活动：{first.campaign.campaign_name}",
                    trigger_moment="基于历史数据分析",
                    user_motivation=first.campaign.target_persona,
                    expected_outcome="参考历史活动效果"
                )
            )
            self._save_memory(memory)
        
        logger.info(f"[Persona Generation Complete] Session: {sid}, Generated {len(campaigns_with_personas)} personas")
        
        return result


# ============ 使用示例 ============

if __name__ == "__main__":
    from goal_determination_node import BrandProfile
    
    # 创建节点
    node = PersonaGenerationNode(llm=None)
    
    # 品牌信息
    brand = BrandProfile(
        brand_name="新茶饮品牌",
        core_appeal="提升知名度",
        is_new_brand=True
    )
    
    # 生成画像
    result = node.run(
        brand_profile=brand,
        primary_goal="acquire_user",
        scene_hint="线下门店打卡",
        persona_hint="年轻人"
    )
    
    print("\n" + "=" * 70)
    print("节点2输出：5个相似活动及其用户画像")
    print("=" * 70)
    
    for i, cp in enumerate(result.campaigns_with_personas, 1):
        print(f"\n{'─' * 70}")
        print(f"【活动 {i}】{cp.campaign.campaign_name}")
        print(f"{'─' * 70}")
        print(f"  相似度: {cp.campaign.performance.get('similarity_score', 0):.2f}")
        print(f"  参与人数: {cp.campaign.performance.get('participants', 0)}")
        print(f"  UGC数: {cp.campaign.performance.get('ugc_count', 0)}")
        print(f"  转化率: {cp.campaign.performance.get('conversion_rate', 0):.1%}")
        
        print(f"\n  【用户画像】{cp.persona_card.persona_name}")
        print(f"  核心特征: {', '.join(cp.persona_card.core_traits)}")
        print(f"  痛点: {', '.join(cp.persona_card.pain_points[:3])}")
        print(f"  爽点: {', '.join(cp.persona_card.pleasure_points[:3])}")
        
        # 显示统计数据
        if cp.data_stats:
            demo = cp.data_stats.get("demographics", {})
            behavior = cp.data_stats.get("behavior", {})
            print(f"\n  【数据统计】")
            print(f"    年龄分布: {demo.get('age_distribution', {})}")
            print(f"    性别比例: {demo.get('gender_ratio', {})}")
            print(f"    渠道偏好: {behavior.get('channel_preferences', {})}")
    
    print(f"\n{'=' * 70}")
    print(f"总计生成 {len(result.campaigns_with_personas)} 个活动画像")
    print(f"{'=' * 70}")
