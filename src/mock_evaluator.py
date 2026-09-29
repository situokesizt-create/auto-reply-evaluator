"""Transparent deterministic rule fixture, not a semantic quality judge."""
import json

from .config import METRICS
from .data_loader import Case
from .parser import parse_judgment


def evaluate_mock(case: Case, criteria: str = "") -> dict:
    """Use only visible text features; no ID lookup, reference labels or randomness."""
    reply, question = case.auto_reply, case.question
    asks = any(s in reply for s in ("请问", "提供", "告诉我"))
    assists = any(s in reply for s in ("帮您", "协助", "跟进"))
    deflects = any(s in reply for s in ("自行", "查看", "联系客服", "耐心等待"))
    empathy = any(s in reply for s in ("抱歉", "感谢", "理解"))
    distressed = any(s in question for s in ("太差", "没人", "坏的", "不工作", "敏感", "取不出来", "搞不懂", "不知道", "异地"))
    actionable = any(s in reply for s in ("申请", "点击", "订单号", "密码", "配对", "检查"))
    unsupported = any(s in reply for s in ("自动提醒", "补偿优惠券"))
    contradictory = "先退货再重新下单" in reply and "会发出新" in reply
    scores = {"correctness": 2 if contradictory else 3 if unsupported else 4,
              "relevance": 3 if "可能" in reply or (distressed and not asks) else 4,
              "completeness": min(5, 2 + int(actionable) + int(asks)),
              "clarity": 2 if contradictory else 3 if len(reply) > 130 else 5,
              "service": max(1, min(5, 2 + int(empathy) + int(assists) + int(asks)
                                      - int(deflects) - int(distressed and not empathy)))}
    details = {
        "correctness": "检测到退款重下单与收到后发新货两种流程混用" if contradictory else
                       "检测到需核实的提醒/补偿承诺" if unsupported else "规则未发现预定义风险；不能据此证实事实正确",
        "relevance": f"通用可能性表述={'可能' in reply}，困难场景缺少追问={distressed and not asks}；未进行语义判定",
        "completeness": f"操作词命中={actionable}，追问词命中={asks}；关键词不代表实际信息完整",
        "clarity": f"回复长度={len(reply)}，预定义流程冲突={contradictory}",
        "service": f"礼貌/安抚词={empathy}，协助词={assists}，追问词={asks}，转介/自助词={deflects}",
    }
    weakest = min(scores, key=scores.get)
    if contradictory:
        improvement = "核实并统一换货与退货重下单流程，追问两件商品分别需要的尺码；运费承担以实际政策为准。"
    elif unsupported:
        improvement = "先核实提醒或补偿功能是否实际支持，避免无依据承诺；追问具体商品/订单，再给出可执行的后续安排。"
    elif not asks:
        improvement = "追问完成当前任务所缺少的商品、订单或具体困难；在实际权限内明确协助方式，避免仅重复通用流程。"
    else:
        improvement = "基于已追问的信息进一步提供针对性下一步，核实事实与服务能力，并回应用户情绪。"
    data = {key: {"score": scores[key], "reason": "[MOCK规则] " + details[key]} for key in METRICS}
    data.update(overall_reason=f"[MOCK演示] 最低维度为{METRICS[weakest]['label']}；{details[weakest]}。不构成真实质量判断。",
                improvement="[MOCK规则建议] " + improvement)
    return parse_judgment(json.dumps(data, ensure_ascii=False))
