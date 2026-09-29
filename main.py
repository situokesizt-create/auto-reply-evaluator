"""CLI entry point: python main.py --mode mock|api [--limit 3]."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import logging
from pathlib import Path
import sys

from src.analyzer import analyze, terminal_summary
from src.config import METRICS, ROOT, RUBRIC_VERSION, Settings
from src.data_loader import load_data
from src.evaluator import DeepSeekJudge, JUDGE_SYSTEM, scoring_payload
from src.mock_evaluator import evaluate_mock
from src.reporter import create_report, save_results, write_json
from src.validator import AUDIT_SYSTEM, summarize_validation, validate_case
from src.visualizer import create_charts


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def runtime_versions() -> dict[str, str]:
    """Record the relevant runtime without dumping potentially secret environment data."""
    versions = {"python": sys.version.split()[0]}
    for name in ("openai", "python-dotenv", "pandas", "matplotlib", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run(argv: list[str] | None = None) -> int:
    """Evaluate independently, persist errors and exit nonzero on partial failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "api"), default="mock")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
        if args.mode == "api" and (not settings.api_key or settings.api_key == "your_api_key_here"):
            print("DEEPSEEK_API_KEY is not configured.\nPlease copy .env.example to .env and configure your API key.", file=sys.stderr)
            return 2
        cases, criteria, warnings = load_data(args.data_dir.resolve())
        cases = cases[:args.limit]
        output = args.output_dir.resolve()
        # Never let artifact destinations overwrite source directories.
        source_dir = args.data_dir.resolve()
        if output == source_dir:
            raise ValueError("Output directory must differ from data directory")
        output.mkdir(parents=True, exist_ok=True)
        judge = DeepSeekJudge(settings) if args.mode == "api" else None
        metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
                    "model": settings.model if judge else "deterministic-rules-v1",
                    "base_url": settings.base_url if judge else None,
                    "temperature": 0 if judge else None, "max_attempts": settings.attempts,
                    "rubric_version": RUBRIC_VERSION, "rubric": METRICS,
                    "criteria_sha256": fingerprint(criteria),
                    "dataset_sha256": fingerprint([vars(case) for case in cases]),
                    "scoring_prompt_sha256": fingerprint(JUDGE_SYSTEM),
                    "audit_prompt_sha256": fingerprint(AUDIT_SYSTEM),
                    "selected_cases": len(cases), "data_warnings": warnings}
        metadata["runtime_versions"] = runtime_versions()
        results, audits = [], []
        for index, case in enumerate(cases, 1):
            logging.info("[%d/%d] evaluating %s...", index, len(cases), case.id)
            result = {"id": case.id, "mode": args.mode, "question": case.question,
                      "auto_reply": case.auto_reply,
                      "input_sha256": fingerprint(scoring_payload(case, criteria))}
            try:
                result.update(judge.evaluate(case, criteria) if judge else evaluate_mock(case, criteria))
                result["error"] = None
            except Exception as exc:
                # API transport exposes only sanitized error types, not secrets.
                result.update(scores={}, reasons={}, overall_score=None, score_100=None,
                              error=str(exc), overall_reason="评分失败", improvement="排查失败后重新评估")
            if judge:
                result["trace"] = judge.trace.copy()
            results.append(result)
            audits.append(validate_case(case, result, args.mode, judge))
            # Incremental checkpoint for long API runs and user interruptions.
            write_json(output / "checkpoint.json", {"metadata": metadata, "results": results, "audits": audits})
        summary = analyze(results)
        validation = summarize_validation(audits)
        save_results(output, results)
        write_json(output / "summary.json", summary)
        write_json(output / "validation_results.json", {"summary": validation, "cases": audits})
        write_json(output / "run_metadata.json", metadata)
        create_charts(results, summary, validation, output, args.mode)
        create_report(output, results, summary, audits, validation, metadata)
        print(terminal_summary(summary, args.mode))
        print(f"Human annotation audit: {validation['assessed']}/{validation['total']} assessed (not accuracy)")
        if validation["assessed"]:
            print(
                f"Audit evidence verified: {validation['evidence_verified']}/{validation['assessed']} "
                "(verbatim provenance check)"
            )
        print(f"Artifacts: {output}")
        return 1 if summary["failed"] or validation["counts"]["error"] else 0
    except (ValueError, FileNotFoundError, ImportError) as exc:
        logging.error("Configuration/input error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
