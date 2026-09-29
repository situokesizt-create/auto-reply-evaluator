"""Validate the actual source schema and join by ID, never by list position."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    auto_reply: str
    reference: str | None = None
    notes: str | None = None


def _rows(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        raw = next((raw[k] for k in ("cases", "data", "items") if k in raw), raw)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError(f"{path.name}: expected a list of objects")
    seen = set()
    for row in raw:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"{path.name}: missing, invalid or duplicate id: {case_id!r}")
        seen.add(case_id)
    return raw


def _field(row: dict, *names: str, required: bool = True) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if required:
        raise ValueError(f"{row.get('id')}: missing nonempty string {names}")
    return None


def load_data(directory: Path) -> tuple[list[Case], str, list[str]]:
    """Support real filenames plus original task3_ names and common field aliases."""
    def locate(name: str) -> Path:
        for folder in (directory, directory.parent):
            for filename in (name, f"task3_{name}"):
                candidate = folder / filename
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"Missing {name} in {directory} or its parent")

    replies = _rows(locate("auto_replies.json"))
    references = {r["id"]: r for r in _rows(locate("human_ref.json"))}
    warnings = []
    cases = []
    for row in replies:
        ref = references.get(row["id"], {})
        if not ref:
            warnings.append(f"{row['id']}: no matching human reference")
        cases.append(Case(row["id"], _field(row, "user_question", "question"),
                          _field(row, "auto_reply", "reply"),
                          _field(ref, "human_reference", "reference", required=False),
                          _field(ref, "annotator_notes", "notes", required=False)))
    orphan_ids = sorted(set(references) - {c.id for c in cases})
    if orphan_ids:
        warnings.append(f"Unmatched reference IDs: {orphan_ids}")
    criteria = locate("eval_criteria.md").read_text(encoding="utf-8-sig")
    if not cases or not criteria.strip():
        raise ValueError("Cases and criteria must be nonempty")
    return cases, criteria, warnings
