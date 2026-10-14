"""Build and synchronize an Opik Test Suite from o11y-bench task specs."""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from grading.models import Problem

from .config import DEFAULT_N_ATTEMPTS, ROOT

DEFAULT_TEST_SUITE_NAME = "o11y-bench"
DEFAULT_SPECS_DIR = ROOT / "tasks-spec"
_DEFAULT_PASS_THRESHOLD = 1
_TEST_CASE_NAMESPACE = uuid.UUID("33af940f-9b80-5ca2-a64e-a62bdd13ef51")


def test_case_uuid(task_id: str) -> str:
    """Return the stable Opik dataset-item UUID for an o11y-bench task."""

    return str(uuid.uuid5(_TEST_CASE_NAMESPACE, task_id))


def _suite_name() -> str:
    return (
        os.getenv("OPIK_TEST_SUITE_NAME", DEFAULT_TEST_SUITE_NAME).strip()
        or DEFAULT_TEST_SUITE_NAME
    )


def _execution_policy() -> dict[str, int]:
    runs = int(os.getenv("OPIK_RUNS_PER_ITEM", str(DEFAULT_N_ATTEMPTS)))
    threshold = int(os.getenv("OPIK_PASS_THRESHOLD", str(_DEFAULT_PASS_THRESHOLD)))
    if runs < 1:
        raise ValueError("OPIK_RUNS_PER_ITEM must be at least 1")
    if threshold < 1 or threshold > runs:
        raise ValueError("OPIK_PASS_THRESHOLD must be between 1 and OPIK_RUNS_PER_ITEM")
    return {"runs_per_item": runs, "pass_threshold": threshold}


def build_test_suite_items(specs_dir: Path = DEFAULT_SPECS_DIR) -> list[dict[str, Any]]:
    """Validate task YAML and convert it to Opik Test Suite items."""

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(specs_dir.rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"task spec must be a mapping: {path}")
        problem = Problem(**payload)
        if problem.id in seen:
            raise ValueError(f"duplicate task id {problem.id!r}: {path}")
        seen.add(problem.id)

        try:
            source_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            source_path = path.relative_to(specs_dir).as_posix()

        items.append(
            {
                "id": test_case_uuid(problem.id),
                "data": {
                    "task_id": problem.id,
                    "statement": problem.statement,
                    "category": problem.category,
                    "tags": problem.tags,
                    "source_path": source_path,
                },
                "description": problem.statement,
                "assertions": [rubric.criterion for rubric in problem.rubric],
            }
        )

    if not items:
        raise ValueError(f"no task specs found under {specs_dir}")
    return items


def _comparable_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id") or ""),
        "data": item.get("data") or {},
        "description": item.get("description") or None,
        "assertions": item.get("assertions") or [],
    }


def sync_test_suite(
    *,
    specs_dir: Path = DEFAULT_SPECS_DIR,
    prune: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Any:
    """Synchronize source-controlled tasks into an Opik Test Suite."""

    opik = _import_opik()
    items = build_test_suite_items(specs_dir)
    policy = _execution_policy()
    project_name = os.getenv("OPIK_PROJECT_NAME", "").strip() or None

    if progress is not None:
        progress(f"[1/3] Loaded and validated {len(items)} task specs from {specs_dir}")

    client = opik.Opik()
    suite = client.get_or_create_test_suite(
        name=_suite_name(),
        description="Observability and SRE agent benchmark maintained in o11y-bench task YAML.",
        global_assertions=[],
        global_execution_policy=policy,
        tags=["o11y-bench", "observability", "sre"],
        project_name=project_name,
    )
    suite.update_test_settings(
        global_assertions=[],
        global_execution_policy=policy,
    )

    if progress is not None:
        progress(f"[2/3] Reconciling Opik Test Suite {_suite_name()!r}")

    existing = {str(item["id"]): item for item in suite.get_items()}
    desired = {str(item["id"]): item for item in items}
    new_items = [item for item_id, item in desired.items() if item_id not in existing]
    changed_items = [
        item
        for item_id, item in desired.items()
        if item_id in existing and _comparable_item(existing[item_id]) != _comparable_item(item)
    ]
    stale_ids = sorted(set(existing) - set(desired)) if prune else []

    if new_items:
        suite.insert(new_items, deduplication=False)
    if changed_items:
        suite.update(changed_items, deduplication=False)
    if stale_ids:
        suite.delete(stale_ids)

    if progress is not None:
        progress(
            f"[3/3] Synchronized {len(items)} cases "
            f"({len(new_items)} added, {len(changed_items)} updated, {len(stale_ids)} removed)"
        )
    client.flush()
    return suite


def opik_url() -> str:
    """Return the configured Opik UI base URL."""

    raw = (
        os.getenv("OPIK_URL_OVERRIDE", "").strip()
        or os.getenv("OPIK_BASE_URL", "").strip()
        or "https://www.comet.com/opik/api"
    )
    return raw.rstrip("/").removesuffix("/api")


def _import_opik() -> Any:
    try:
        import opik
    except ImportError as exc:
        raise RuntimeError(
            "Opik publishing requires opik==2.2.100. "
            "Run `mise run opik:setup` to install the optional integration."
        ) from exc
    return opik


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize o11y-bench task specs to an Opik Test Suite"
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_SPECS_DIR)
    parser.add_argument("--no-prune", action="store_true")
    args = parser.parse_args()

    suite = sync_test_suite(
        specs_dir=args.path,
        prune=not args.no_prune,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Test Suite: {suite.name} ({suite.items_count} cases)")
    print(f"Open Opik: {opik_url()}")


if __name__ == "__main__":
    main()
