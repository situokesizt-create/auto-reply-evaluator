"""UTF-8 artifacts and a self-contained, traceable Markdown report."""
import json
from pathlib import Path

import pandas as pd

from .config import LIMITATIONS, METRICS


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _quote(value: str) -> str:
    return "\n".join("> " + line for line in value.splitlines())


def save_results(output: Path, results: list[dict]) -> None:
    """Checkpoint final case results before plotting so chart failure loses no scores."""
    write_json(output / "evaluation_results.json", results)
    rows = []
    for result in results:
        row = {k: result.get(k) for k in ("id", "mode", "question", "auto_reply", "overall_score", "score_100", "overall_reason", "improvement", "error")}
        for key in METRICS:
            row[key] = result.get("scores", {}).get(key)
            row[f"{key}_reason"] = result.get("reasons", {}).get(key)
        # Prevent spreadsheet formula execution when users open an untrusted dataset.
        rows.append({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v for k, v in row.items()})
    pd.DataFrame(rows).to_csv(output / "evaluation_results.csv", index=False, encoding="utf-8-sig")


def create_report(output: Path, results: list[dict], summary: dict,
                  audits: list[dict], validation: dict, metadata: dict) -> None:
    """Generate all findings from actual results, with explicit mock/API provenance."""
    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f}"
    is_mock = metadata["mode"] == "mock"
    lines = ["# 自动回复质量评估报告", "",
             "> MOCK 演示：以下得分来自确定性规则，仅证明流水线可运行，不能用于判断上线质量。" if is_mock else
             "> DeepSeek API 实际调用结果；人工对比为模型辅助定性审计，不是独立人工准确率。", "",
             "## 1. 评估概览", "",
             f"- 运行时间：{metadata['created_at']}；模式：{metadata['mode']}；模型：{metadata['model']}",
             f"- Rubric：{metadata['rubric_version']}；完整配置、输入与 Prompt 哈希见 `run_metadata.json`。",
             f"- 样本数量：{summary['total']}；评分成功：{summary['successful']}；失败：{summary['failed']}。",
             f"- 平均总分：{fmt(summary['average'])}/5；中位数：{fmt(summary['median'])}；百分制均分：{fmt(summary['score_100'])}。",
             f"- 最高分：{fmt(summary['best'])}；最低分：{fmt(summary['worst'])}。失败不按零分计算。", "",
             "## 2. 从业务要求到指标", "",
             "“准确、不瞎编”合并为正确性与事实依据；“有用”拆为相关性和完整性与可操作性；“语气好”单列同理心与服务意识，同时保留清晰度。人工注释帮助明确主动协助的重要性，但不自动视为金标准。", "",
             "| 指标 | 权重 | 均分 | 最低 | 最高 |", "|---|---:|---:|---:|---:|"]
    for key, spec in METRICS.items():
        stat = summary["metrics"].get(key, {})
        lines.append(f"| {spec['label']} | {spec['weight']:.0%} | {fmt(stat.get('average'))} | {fmt(stat.get('min'))} | {fmt(stat.get('max'))} |")
    lines += ["", "## 3. 得分分布", "", "![各指标均分与分布](metric_distribution.png)", "",
              "![逐条总分](overall_scores.png)", "", "## 4. 最差 3 条 Case", ""]
    for result in summary["lowest"]:
        lines += [f"### {result['id']} · {result['overall_score']:.2f}/5", "", "用户问题：", "", _quote(result["question"]), "",
                  "自动回复：", "", _quote(result["auto_reply"]), "", "| 指标 | 得分 | 理由 |", "|---|---:|---|"]
        for key in METRICS:
            lines.append(f"| {METRICS[key]['label']} | {result['scores'][key]} | {_cell(result['reasons'][key])} |")
        lines += ["", f"主要问题：{result['overall_reason']}", "", f"改进建议：{result['improvement']}", ""]
    if not summary["lowest"]:
        lines += ["没有成功评分，无法排序。", ""]
    lines += ["## 5. 与人工评价对比", "",
              "源文件只有 `human_reference` 和 `annotator_notes`，没有人工分数或通过/失败标签，因此不计算 Pearson、Spearman、MAE、Precision、Recall、F1 或混淆矩阵。", "",
              "API 模式下先评分（不读取 annotator_notes），再独立发起定性对比请求。核验审计引文确实来自原回复和人工分析，并保留分歧及标注疑点。", "",
              f"审计覆盖：{validation['assessed']}/{validation['total']}（{validation['coverage']:.0%}）；{validation['interpretation']}", "",
              "![人工分析审计覆盖](validation.png)", "",
              "| Case | 结论 | 一致/分歧原因 | 人工标注疑点 |", "|---|---|---|---|"]
    for audit in audits:
        lines.append(f"| {audit['id']} | {audit['status']} | {_cell(audit['explanation'])} | {_cell(audit.get('annotation_caveat', '尚未审计'))} |")
    lines += ["", "完整人工分析、引用证据及请求记录见 `validation_results.json`。", "",
              "## 6. 主要发现", ""]
    if summary["metrics"]:
        worst_key = min(summary["metrics"], key=lambda k: summary["metrics"][k]["average"])
        lines += [f"- 本次{'演示规则' if is_mock else 'Judge'}中均分最低的是{METRICS[worst_key]['label']}（{summary['metrics'][worst_key]['average']:.2f}）。",
                  f"- 最低三个样本为 {', '.join(r['id'] for r in summary['lowest'])}，需结合原文复核，不能仅凭总分下结论。"]
    lines += [f"- 评分失败 {summary['failed']} 条；审计失败 {validation['counts']['error']} 条；未审计 {validation['counts']['not_assessed']} 条。",
              "- mock 不评价语义一致性，不从规则得分推断上线质量。" if is_mock else
              f"- 定性对比：一致 {validation['counts']['agree']}，部分一致 {validation['counts']['partial']}，不一致 {validation['counts']['disagree']}，证据不足 {validation['counts']['uncertain']}。",
              "", "## 7. 方法局限性", "", LIMITATIONS, "## 8. 改进方向", "",
              "建立经过双人复核的独立标注集，先定义评分锚点和标签，再评估相关性与分类指标。补充企业事实证据和工具能力说明，开展多 Judge、重复评估及按场景分层的 CI 回归。", "",
              "## 9. 失败与数据警告", ""]
    issues = metadata.get("data_warnings", []) + [f"{r['id']}: {r['error']}" for r in results if r.get("error")]
    issues += [f"{a['id']} audit: {a['error']}" for a in audits if a.get("error")]
    lines += [f"- {issue}" for issue in issues] if issues else ["无运行失败或数据关联警告。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
