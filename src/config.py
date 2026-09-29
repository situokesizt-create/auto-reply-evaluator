"""Configuration and the versioned business rubric (single source of truth)."""
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RUBRIC_VERSION = "1.0"
METRICS = {
    "correctness": {"label": "正确性与事实依据", "weight": 0.30,
        "definition": "事实、政策、数字、步骤是否与证据一致；避免捏造商品、订单状态、服务能力及补偿承诺。未知不等于错误，注明证据不足。",
        "anchors": ["核心事实严重错误或严重编造", "核心内容错误", "明显错误或高风险无依据承诺", "部分准确但有信息偏差或依据不足", "基本准确，有轻微不严谨", "已有证据支持，无明显错误且妥善处理未知"]},
    "relevance": {"label": "相关性", "weight": 0.15,
        "definition": "是否回应实际诉求，正确理解具体商品、订单、多意图问题，避免模板化转移话题。",
        "anchors": ["完全无关", "基本答非所问", "较多无关内容", "只回答部分诉求", "基本相关，少量冗余", "直接回应全部实际诉求"]},
    "completeness": {"label": "完整性与可操作性", "weight": 0.25,
        "definition": "是否覆盖必要条件、步骤、限制及下一步；信息不足时有效追问。合理自助路径可得高分，不要求虚构代操作能力。",
        "anchors": ["没有有用信息", "仅覆盖极少部分", "缺少多项关键信息或无有效下一步", "遗漏重要信息或下一步不具体", "仅非关键遗漏", "必要信息完整，下一步明确可执行"]},
    "clarity": {"label": "表达清晰度", "weight": 0.10,
        "definition": "语言清晰、简洁、专业、结构合理，无歧义或自相矛盾。",
        "anchors": ["无法理解", "难以理解", "明显歧义或混乱", "可理解但冗长或不清楚", "整体清晰，轻微表达问题", "清晰简洁，专业自然"]},
    "service": {"label": "同理心与服务意识", "weight": 0.20,
        "definition": "语气尊重，识别焦虑或不满，主动承担合理协助和跟进责任，减少不必要操作负担。不得为了主动而承诺不存在的能力。",
        "anchors": ["冒犯或恶意推诿", "冷漠且完全推诿", "明显缺乏场景关怀或加重负担", "礼貌但模板化、主动协助不足", "有针对性关怀与合理协助", "充分回应情绪和诉求，主动且可兑现地协助"]},
}
WEIGHTS = {key: float(value["weight"]) for key, value in METRICS.items()}

LIMITATIONS = """- LLM Judge 会受模型版本、Prompt、temperature、评价顺序影响；低温和固定 Prompt 仍不保证确定性，可多次评分取均值或多 Judge 投票。
- 参考答案并不唯一；按语义和事实评估，不能用字符串相似度代替质量判断。
- 缺乏企业政策、商品库、订单状态及工具能力证据；参考回复中的期限、金额与 XX 占位符不是权威事实，应接入知识库开展有依据的事实核验。
- 五项指标权重具有业务主观性；主动服务与可操作性存在相关性，需避免重复扣分，并通过满意度、投诉率及人工质检校准。
- 仅 20 条样本，不能代表线上流量；需要扩充样本、按问题类型分层，并建立回归评测集。
- 人工只有文字分析，没有数值或二元标签；定性审计不是准确率。评分使用人工参考，验证使用同批人工分析，也不是独立测试集。同一模型的审计还存在自我偏好。
- 人工分析可能与原回复冲突；不能盲从。主动服务不等于声称已经查过订单、能代操作或一定能发补偿。
"""

@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0
    attempts: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        """Read only this project's .env, without overriding shell variables."""
        load_dotenv(ROOT / ".env")
        return cls(os.getenv("DEEPSEEK_API_KEY", "").strip(),
                   os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                   os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
