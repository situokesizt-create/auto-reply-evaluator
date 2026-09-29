"""Offline contract, failure isolation and integration tests."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main
from src.analyzer import analyze
from src.config import ROOT, Settings, WEIGHTS
from src.data_loader import Case, load_data
from src.evaluator import DeepSeekJudge, scoring_payload
from src.mock_evaluator import evaluate_mock
from src.parser import extract_json, parse_judgment, weighted_score
from src.validator import parse_audit, summarize_validation, validate_case


def judgment(score: object = 4) -> dict:
    return {**{key: {"score": score, "reason": "具体证据"} for key in WEIGHTS},
            "overall_reason": "整体原因", "improvement": "可执行建议"}


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Here is the result:\n{}\nEnd."])
def test_json_wrappers(wrapper: str) -> None:
    parsed = parse_judgment(wrapper.format(json.dumps(judgment())))
    assert parsed["overall_score"] == 4


def test_numeric_conversion_clamping_and_local_total() -> None:
    data = judgment("4")
    data["correctness"]["score"] = 9
    data["clarity"]["score"] = -3
    data["overall_score"] = 1000
    parsed = parse_judgment(json.dumps(data))
    assert parsed["scores"]["correctness"] == 5
    assert parsed["scores"]["clarity"] == 0
    assert parsed["overall_score"] == 3.9
    assert len(parsed["warnings"]) == 5


@pytest.mark.parametrize("score", [True, None, [], "abc", "NaN", "Infinity", "-Infinity"])
def test_invalid_scores_rejected(score: object) -> None:
    with pytest.raises(ValueError):
        parse_judgment(json.dumps(judgment(score)))


@pytest.mark.parametrize("text", ["", "not json", "{} {}", "{broken", "{}"])
def test_invalid_schema_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_judgment(text)


def test_weighted_score() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1)
    assert weighted_score(dict.fromkeys(WEIGHTS, 5)) == 5
    assert weighted_score(dict.fromkeys(WEIGHTS, 0)) == 0
    assert weighted_score(dict(correctness=2, relevance=4, completeness=3, clarity=5, service=1)) == 2.65


def test_lowest_sort_excludes_failure() -> None:
    rows = [{"id": str(i), **parse_judgment(json.dumps(judgment(score)))}
            for i, score in enumerate([4, 2, 5, 1, 3])]
    rows.append({"id": "failed", "overall_score": None, "error": "timeout"})
    summary = analyze(rows)
    assert [r["id"] for r in summary["lowest"]] == ["3", "1", "4"]
    assert summary["average"] == 3
    assert summary["failed"] == 1
    assert analyze([rows[-1]])["average"] is None


def test_real_data_mock_and_no_label_leakage() -> None:
    cases, criteria, warnings = load_data(ROOT / "data")
    assert len(cases) == 20 and not warnings
    scores = []
    for case in cases:
        result = evaluate_mock(case)
        assert result == evaluate_mock(case)
        assert 0 <= result["overall_score"] <= 5
        assert set(result["scores"]) == set(WEIGHTS)
        payload = json.dumps(scoring_payload(case, criteria), ensure_ascii=False)
        assert case.notes not in payload
        scores.append(result["overall_score"])
    assert len(set(scores)) > 3


def test_mock_does_not_invent_human_labels() -> None:
    case = Case("a", "问题", "回复", "参考", "人工文字")
    audit = validate_case(case, evaluate_mock(case), "mock", None)
    assert audit["status"] == "not_assessed"
    summary = summarize_validation([audit])
    assert summary["coverage"] == 0
    assert "accuracy" not in summary


def test_audit_tracks_evidence_verification() -> None:
    case = Case(
        "a",
        "问题",
        "您好，请提供订单号",
        "参考",
        "缺少具体信息",
    )

    audit = dict(
        status="partial",
        explanation="理由",
        human_evidence="缺少",
        reply_evidence="订单号",
        annotation_caveat="未发现",
    )

    result = parse_audit(json.dumps(audit), case)

    assert result["status"] == "partial"
    assert result["human_evidence_verified"] is True
    assert result["reply_evidence_verified"] is True
    assert result["evidence_verified"] is True

    audit["reply_evidence"] = "伪造的原文"

    result = parse_audit(json.dumps(audit), case)

    assert result["status"] == "partial"
    assert result["human_evidence_verified"] is True
    assert result["reply_evidence_verified"] is False
    assert result["evidence_verified"] is False
    assert result["warnings"]


def response(content: str) -> SimpleNamespace:
    return SimpleNamespace(usage=None, model="test-model", choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content))])


def test_api_transport_retry_and_json_mode() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [TimeoutError(), response("invalid"), response(json.dumps(judgment()))]
    sleep = Mock()
    judge = DeepSeekJudge(Settings("test-only", "https://example.invalid", "model"), client, sleep)
    result = judge.evaluate(Case("a", "问", "答"), "标准")
    assert result["overall_score"] == 4
    assert client.chat.completions.create.call_count == 3
    assert sleep.call_count == 2
    assert client.chat.completions.create.call_args.kwargs["response_format"] == {"type": "json_object"}
    assert judge.trace["attempts"] == 3


def test_permanent_errors_not_retried_and_secrets_not_logged() -> None:
    client = Mock()
    error = RuntimeError("secret-token")
    error.status_code = 401
    client.chat.completions.create.side_effect = error
    judge = DeepSeekJudge(Settings("test-only", "url", "model"), client, Mock())
    with pytest.raises(RuntimeError, match="HTTP 401") as caught:
        judge.evaluate(Case("a", "问", "答"), "标准")
    assert "secret-token" not in str(caught.value)
    assert client.chat.completions.create.call_count == 1


def test_rate_limit_exhausts_exactly_three_attempts() -> None:
    client = Mock()
    error = RuntimeError("provider details are private")
    error.status_code = 429
    client.chat.completions.create.side_effect = error
    sleep = Mock()
    judge = DeepSeekJudge(Settings("test-only", "url", "model"), client, sleep)
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        judge.evaluate(Case("a", "问", "答"), "标准")
    assert client.chat.completions.create.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


def test_real_sdk_with_offline_http_transport() -> None:
    """Exercise SDK serialization and response parsing without live credentials."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    requests = []
    case = Case("a", "问题", "请提供订单号，我帮您查", "参考", "缺少具体信息")
    audit = dict(status="partial", explanation="追问有帮助但尚未解决", human_evidence="缺少具体信息",
                 reply_evidence="请提供订单号", annotation_caveat="需核实服务能力")
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        content = judgment() if len(requests) == 1 else audit
        return httpx.Response(200, json={"id": "offline-test", "object": "chat.completion",
            "created": 1, "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(content, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
    with openai.OpenAI(api_key="offline-test", base_url="https://offline.invalid", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        judge = DeepSeekJudge(Settings("offline-test", "https://offline.invalid", "test-model"), client)
        result = judge.evaluate(case, "业务标准")
        validation = validate_case(case, result, "api", judge)
    assert result["overall_score"] == 4
    assert validation["status"] == "partial"
    assert len(requests) == 2
    assert validation["trace"]["usage"][0]["total_tokens"] == 30
    assert "annotator_notes" not in requests[0]["messages"][1]["content"]


def test_input_aliases_missing_reference_and_duplicate_id(tmp_path: Path) -> None:
    replies = [{"id": "x", "question": "问", "reply": "答"}]
    (tmp_path / "auto_replies.json").write_text(json.dumps({"cases": replies}), encoding="utf-8")
    (tmp_path / "human_ref.json").write_text("[]", encoding="utf-8")
    (tmp_path / "eval_criteria.md").write_text("业务标准", encoding="utf-8")
    cases, _, warnings = load_data(tmp_path)
    assert cases[0].reference is None and warnings
    (tmp_path / "auto_replies.json").write_text(json.dumps(replies * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_data(tmp_path)


def test_pipeline_failure_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_mock = main.evaluate_mock
    def fails_one(case: Case, criteria: str) -> dict:
        if case.id == "case_01":
            raise RuntimeError("injected failure")
        return real_mock(case, criteria)
    monkeypatch.setattr(main, "evaluate_mock", fails_one)
    assert main.run(["--mode", "mock", "--limit", "3", "--output-dir", str(tmp_path)]) == 1
    rows = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))
    assert rows[0]["overall_score"] is None and rows[1]["overall_score"] is not None
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "validation.png").exists()


def test_all_failed_pipeline_still_creates_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "evaluate_mock", Mock(side_effect=RuntimeError("failure")))
    assert main.run(["--limit", "1", "--output-dir", str(tmp_path)]) == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["average"] is None and summary["lowest"] == []


def test_missing_key_creates_no_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(main.Settings, "from_env", lambda: Settings("", "url", "model"))
    assert main.run(["--mode", "api", "--output-dir", str(tmp_path)]) == 2
    assert "DEEPSEEK_API_KEY is not configured" in capsys.readouterr().err
    assert not (tmp_path / "evaluation_results.json").exists()
