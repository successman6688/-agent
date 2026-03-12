"""
MCP 数据分析工具

提供 SQL 查询和统计分析功能，用于从真实数据中提取用户画像统计值。
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)


class DemographicStats(BaseModel):
    """人口统计统计值"""
    age_distribution: Dict[str, float] = Field(description="年龄分布百分比")
    gender_ratio: Dict[str, float] = Field(description="性别比例")
    income_distribution: Dict[str, float] = Field(description="收入分布")
    city_tier_distribution: Dict[str, float] = Field(description="城市线级分布")


class BehaviorStats(BaseModel):
    """行为偏好统计值"""
    channel_preferences: Dict[str, float] = Field(description="渠道偏好百分比")
    content_preferences: Dict[str, float] = Field(description="内容偏好百分比")
    active_time_distribution: Dict[str, float] = Field(description="活跃时段分布")
    avg_session_duration: float = Field(description="平均会话时长(分钟)")
    engagement_rate: float = Field(description="互动率")


class PainPleasureAnalysis(BaseModel):
    """痛点爽点分析"""
    top_pain_points: List[Dict[str, Any]] = Field(description="TOP5痛点及提及次数")
    top_pleasure_points: List[Dict[str, Any]] = Field(description="TOP5爽点及提及次数")
    sentiment_score: float = Field(description="情感得分(-1到1)")


class CampaignDataQuery(BaseModel):
    """活动数据查询参数"""
    campaign_id: str = Field(description="活动ID")
    brand_id: str = Field(description="品牌ID")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")


class DataAnalyzer:
    """
    数据分析工具类
    
    提供以下功能：
    1. 向量检索相似活动
    2. 查询活动参与用户数据
    3. 分析用户画像统计值
    4. 提取痛点爽点关键词
    """
    
    def __init__(self, db_connection=None, vector_store=None):
        """
        Args:
            db_connection: 数据库连接（可选）
            vector_store: 向量存储（可选）
        """
        self.db = db_connection
        self.vector_store = vector_store
        logger.info("DataAnalyzer initialized")
    
    def search_similar_campaigns(
        self, 
        scene_hint: str, 
        brand_id: str, 
        primary_goal: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        向量检索相似历史活动
        
        实际实现中应该调用向量数据库（如 Milvus/Pinecone）
        这里使用模拟数据演示
        
        Args:
            scene_hint: 场景关键词
            brand_id: 品牌ID
            primary_goal: 首要目标
            top_k: 返回数量
            
        Returns:
            List[Dict]: 相似活动列表
        """
        logger.info(f"[Vector Search] Scene: {scene_hint}, Brand: {brand_id}, Goal: {primary_goal}")
        
        # 模拟向量检索结果
        mock_campaigns = [
            {
                "campaign_id": "CP001",
                "campaign_name": "夏季新品打卡挑战赛",
                "scene_tags": ["线下门店", "打卡", "拍照", "分享"],
                "target_goal": "acquire_user",
                "participants_count": 5234,
                "ugc_count": 1289,
                "conversion_rate": 0.152,
                "similarity_score": 0.92,
                "date_range": "2024-06-01 ~ 2024-06-30"
            },
            {
                "campaign_id": "CP002",
                "campaign_name": "周末探店福利活动",
                "scene_tags": ["线下门店", "探店", "体验", "社交"],
                "target_goal": "brand_awareness",
                "participants_count": 3456,
                "ugc_count": 876,
                "conversion_rate": 0.098,
                "similarity_score": 0.88,
                "date_range": "2024-07-15 ~ 2024-08-15"
            },
            {
                "campaign_id": "CP003",
                "campaign_name": "新品尝鲜体验官招募",
                "scene_tags": ["新品", "体验", "测评", "分享"],
                "target_goal": "ugc_content",
                "participants_count": 1890,
                "ugc_count": 1567,
                "conversion_rate": 0.203,
                "similarity_score": 0.85,
                "date_range": "2024-08-01 ~ 2024-08-31"
            },
            {
                "campaign_id": "CP004",
                "campaign_name": "国庆门店狂欢节",
                "scene_tags": ["线下门店", "节日", "促销", "聚会"],
                "target_goal": "convert_purchase",
                "participants_count": 8921,
                "ugc_count": 2341,
                "conversion_rate": 0.185,
                "similarity_score": 0.81,
                "date_range": "2024-10-01 ~ 2024-10-07"
            },
            {
                "campaign_id": "CP005",
                "campaign_name": "小红书种草计划",
                "scene_tags": ["社交媒体", "种草", "内容", "分享"],
                "target_goal": "brand_awareness",
                "participants_count": 4567,
                "ugc_count": 3421,
                "conversion_rate": 0.125,
                "similarity_score": 0.78,
                "date_range": "2024-09-01 ~ 2024-09-30"
            }
        ]
        
        # 根据场景关键词过滤和排序
        filtered = []
        for camp in mock_campaigns:
            score = camp["similarity_score"]
            if any(tag in scene_hint for tag in camp["scene_tags"]):
                score += 0.05
            camp["final_score"] = min(score, 0.99)
            filtered.append(camp)
        
        filtered.sort(key=lambda x: x["final_score"], reverse=True)
        return filtered[:top_k]
    
    def get_campaign_participants(self, campaign_id: str) -> List[str]:
        """
        获取活动参与用户ID列表
        
        Args:
            campaign_id: 活动ID
            
        Returns:
            List[str]: 用户ID列表
        """
        logger.info(f"[Query Participants] Campaign: {campaign_id}")
        
        # 模拟查询结果
        participant_counts = {
            "CP001": 5234,
            "CP002": 3456,
            "CP003": 1890,
            "CP004": 8921,
            "CP005": 4567
        }
        
        count = participant_counts.get(campaign_id, 1000)
        return [f"user_{campaign_id}_{i}" for i in range(min(count, 100))]  # 限制100个用于演示
    
    def analyze_demographics(self, user_ids: List[str]) -> DemographicStats:
        """
        分析用户人口统计特征
        
        Args:
            user_ids: 用户ID列表
            
        Returns:
            DemographicStats: 人口统计统计值
        """
        logger.info(f"[Analyze Demographics] Users: {len(user_ids)}")
        
        # 模拟统计分析结果
        return DemographicStats(
            age_distribution={
                "18-24岁": 35.5,
                "25-30岁": 42.3,
                "31-35岁": 18.2,
                "36-40岁": 3.8,
                "40岁以上": 0.2
            },
            gender_ratio={
                "女性": 68.5,
                "男性": 31.5
            },
            income_distribution={
                "5K以下": 12.3,
                "5K-10K": 35.7,
                "10K-20K": 38.5,
                "20K-30K": 11.2,
                "30K以上": 2.3
            },
            city_tier_distribution={
                "一线城市": 45.2,
                "新一线": 28.6,
                "二线城市": 18.3,
                "三线及以下": 7.9
            }
        )
    
    def analyze_behavior_patterns(self, user_ids: List[str]) -> BehaviorStats:
        """
        分析用户行为偏好
        
        Args:
            user_ids: 用户ID列表
            
        Returns:
            BehaviorStats: 行为偏好统计值
        """
        logger.info(f"[Analyze Behavior] Users: {len(user_ids)}")
        
        return BehaviorStats(
            channel_preferences={
                "小红书": 42.5,
                "抖音": 28.3,
                "微信朋友圈": 18.7,
                "微博": 6.2,
                "其他": 4.3
            },
            content_preferences={
                "高颜值图片": 38.5,
                "真实测评": 27.3,
                "优惠活动": 18.9,
                "使用教程": 10.2,
                "品牌故事": 5.1
            },
            active_time_distribution={
                "工作日午休(12:00-14:00)": 25.3,
                "工作日晚间(19:00-22:00)": 35.7,
                "周末上午(10:00-12:00)": 15.2,
                "周末下午(14:00-18:00)": 18.6,
                "深夜(22:00-24:00)": 5.2
            },
            avg_session_duration=8.5,
            engagement_rate=0.342
        )
    
    def extract_pain_pleasure_points(self, campaign_id: str) -> PainPleasureAnalysis:
        """
        从用户反馈中提取痛点和爽点
        
        Args:
            campaign_id: 活动ID
            
        Returns:
            PainPleasureAnalysis: 痛点爽点分析
        """
        logger.info(f"[Extract Pain/Pleasure] Campaign: {campaign_id}")
        
        # 模拟从评论、反馈中提取的关键词分析
        pain_points_map = {
            "CP001": [
                {"point": "排队时间太长", "mentions": 234, "percentage": 18.5},
                {"point": "门店位置不好找", "mentions": 189, "percentage": 14.9},
                {"point": "优惠规则复杂", "mentions": 156, "percentage": 12.3},
                {"point": "新品口味不符合预期", "mentions": 134, "percentage": 10.6},
                {"point": "拍照光线不好", "mentions": 98, "percentage": 7.7}
            ],
            "CP002": [
                {"point": "活动名额有限", "mentions": 198, "percentage": 22.1},
                {"point": "预约流程繁琐", "mentions": 156, "percentage": 17.4},
                {"point": "门店服务响应慢", "mentions": 134, "percentage": 14.9}
            ],
            "default": [
                {"point": "价格偏高", "mentions": 245, "percentage": 20.1},
                {"point": "选择困难", "mentions": 198, "percentage": 16.3},
                {"point": "担心踩雷", "mentions": 167, "percentage": 13.7},
                {"point": "活动规则复杂", "mentions": 134, "percentage": 11.0},
                {"point": "优惠力度不够", "mentions": 112, "percentage": 9.2}
            ]
        }
        
        pleasure_points_map = {
            "CP001": [
                {"point": "拍照超出片", "mentions": 567, "percentage": 32.5},
                {"point": "新品口感惊艳", "mentions": 456, "percentage": 26.1},
                {"point": "店员服务态度好", "mentions": 345, "percentage": 19.8},
                {"point": "优惠很实在", "mentions": 234, "percentage": 13.4},
                {"point": "社交货币价值高", "mentions": 145, "percentage": 8.3}
            ],
            "default": [
                {"point": "独特体验", "mentions": 456, "percentage": 28.5},
                {"point": "社交货币", "mentions": 389, "percentage": 24.3},
                {"point": "身份认同", "mentions": 298, "percentage": 18.6},
                {"point": "品质保证", "mentions": 234, "percentage": 14.6},
                {"point": "惊喜感", "mentions": 178, "percentage": 11.1}
            ]
        }
        
        pain_points = pain_points_map.get(campaign_id, pain_points_map["default"])
        pleasure_points = pleasure_points_map.get(campaign_id, pleasure_points_map["default"])
        
        return PainPleasureAnalysis(
            top_pain_points=pain_points,
            top_pleasure_points=pleasure_points,
            sentiment_score=0.35  # 正面情感
        )
    
    def generate_persona_traits(
        self, 
        demo_stats: DemographicStats,
        behavior_stats: BehaviorStats,
        pain_pleasure: PainPleasureAnalysis
    ) -> List[str]:
        """
        基于统计数据生成核心特征标签
        
        Args:
            demo_stats: 人口统计
            behavior_stats: 行为统计
            pain_pleasure: 痛点爽点分析
            
        Returns:
            List[str]: 特征标签列表
        """
        traits = []
        
        # 基于年龄分布
        age_max = max(demo_stats.age_distribution.items(), key=lambda x: x[1])
        if age_max[0] in ["18-24岁", "25-30岁"]:
            traits.append("年轻活力")
        
        # 基于性别比例
        if demo_stats.gender_ratio.get("女性", 0) > 60:
            traits.append("女性主导")
        
        # 基于城市分布
        tier1 = demo_stats.city_tier_distribution.get("一线城市", 0)
        new_tier1 = demo_stats.city_tier_distribution.get("新一线", 0)
        if tier1 + new_tier1 > 60:
            traits.append("都市白领")
        
        # 基于渠道偏好
        channel_max = max(behavior_stats.channel_preferences.items(), key=lambda x: x[1])
        if channel_max[0] == "小红书":
            traits.append("种草达人")
        elif channel_max[0] == "抖音":
            traits.append("短视频爱好者")
        
        # 基于互动率
        if behavior_stats.engagement_rate > 0.3:
            traits.append("高互动意愿")
        
        # 基于痛点
        top_pain = pain_pleasure.top_pain_points[0]["point"] if pain_pleasure.top_pain_points else ""
        if "排队" in top_pain or "时间" in top_pain:
            traits.append("时间敏感")
        
        return traits[:6]  # 最多6个标签


# ============ 使用示例 ============

if __name__ == "__main__":
    analyzer = DataAnalyzer()
    
    # 1. 搜索相似活动
    campaigns = analyzer.search_similar_campaigns(
        scene_hint="线下门店打卡",
        brand_id="BR001",
        primary_goal="acquire_user"
    )
    
    print("=" * 60)
    print("相似活动检索结果")
    print("=" * 60)
    for i, camp in enumerate(campaigns, 1):
        print(f"\n{i}. {camp['campaign_name']}")
        print(f"   相似度: {camp['similarity_score']:.2f}")
        print(f"   参与人数: {camp['participants_count']}")
        print(f"   UGC数: {camp['ugc_count']}")
        print(f"   转化率: {camp['conversion_rate']:.1%}")
    
    # 2. 分析第一个活动的用户画像
    if campaigns:
        camp_id = campaigns[0]["campaign_id"]
        participants = analyzer.get_campaign_participants(camp_id)
        
        print("\n" + "=" * 60)
        print(f"活动 {camp_id} 用户画像分析")
        print("=" * 60)
        
        # 人口统计
        demo = analyzer.analyze_demographics(participants)
        print(f"\n年龄分布: {demo.age_distribution}")
        print(f"性别比例: {demo.gender_ratio}")
        print(f"收入分布: {demo.income_distribution}")
        
        # 行为偏好
        behavior = analyzer.analyze_behavior_patterns(participants)
        print(f"\n渠道偏好: {behavior.channel_preferences}")
        print(f"活跃时段: {behavior.active_time_distribution}")
        print(f"平均会话时长: {behavior.avg_session_duration}分钟")
        
        # 痛点爽点
        pp = analyzer.extract_pain_pleasure_points(camp_id)
        print(f"\nTOP3痛点:")
        for p in pp.top_pain_points[:3]:
            print(f"  - {p['point']} ({p['percentage']}%)")
        print(f"\nTOP3爽点:")
        for p in pp.top_pleasure_points[:3]:
            print(f"  - {p['point']} ({p['percentage']}%)")
        
        # 生成特征标签
        traits = analyzer.generate_persona_traits(demo, behavior, pp)
        print(f"\n核心特征标签: {traits}")
