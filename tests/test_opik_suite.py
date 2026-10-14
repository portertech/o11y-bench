from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from o11y_bench import opik_suite


class FakeSuite:
    def __init__(self, items=None):
        self.name = "o11y-bench"
        self.items = list(items or [])
        self.settings = []
        self.inserted = []
        self.updated = []
        self.deleted = []

    @property
    def items_count(self):
        return len(self.items)

    def update_test_settings(self, **kwargs):
        self.settings.append(kwargs)

    def get_items(self):
        return list(self.items)

    def insert(self, items, *, deduplication=True):
        assert deduplication is False
        self.inserted.extend(items)
        self.items.extend(items)

    def update(self, items, *, deduplication=True):
        assert deduplication is False
        self.updated.extend(items)
        by_id = {item["id"]: item for item in self.items}
        by_id.update({item["id"]: item for item in items})
        self.items = list(by_id.values())

    def delete(self, item_ids):
        self.deleted.extend(item_ids)
        stale = set(item_ids)
        self.items = [item for item in self.items if item["id"] not in stale]


class FakeClient:
    def __init__(self, suite):
        self.suite = suite
        self.create_kwargs = None
        self.flushed = False

    def get_or_create_test_suite(self, **kwargs):
        self.create_kwargs = kwargs
        return self.suite

    def flush(self):
        self.flushed = True


def _write_spec(tmp_path, *, statement="Investigate the incident."):
    spec = tmp_path / "investigation" / "incident.yaml"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        f"""\
id: incident
category: investigation
tags: [metrics, logs]
statement: {statement}
checks: []
rubric:
  - criterion: The answer identifies the cause.
    weight: 1
"""
    )
    return spec


def test_build_test_suite_items_maps_task_specs_and_stable_id(tmp_path):
    _write_spec(tmp_path)

    items = opik_suite.build_test_suite_items(tmp_path)

    assert len(items) == 1
    item = items[0]
    assert item["id"] == opik_suite.test_case_uuid("incident")
    assert uuid.UUID(item["id"]).version == 7
    assert item["data"]["task_id"] == "incident"
    assert item["data"]["category"] == "investigation"
    assert item["data"]["tags"] == ["metrics", "logs"]
    assert item["assertions"] == ["The answer identifies the cause."]


def test_build_test_suite_items_rejects_duplicate_ids(tmp_path):
    for directory in ("one", "two"):
        path = tmp_path / directory / "task.yaml"
        path.parent.mkdir()
        path.write_text(
            "id: duplicate\ncategory: test\nstatement: Run it.\nchecks: []\n"
            "rubric:\n  - criterion: It works.\n    weight: 1\n"
        )

    with pytest.raises(ValueError, match="duplicate task id"):
        opik_suite.build_test_suite_items(tmp_path)


def test_sync_test_suite_reconciles_changes_and_prunes(monkeypatch, tmp_path):
    _write_spec(tmp_path, statement="Investigate the updated incident.")
    desired_id = opik_suite.test_case_uuid("incident")
    suite = FakeSuite(
        [
            {
                "id": desired_id,
                "data": {"task_id": "incident", "statement": "Old statement"},
                "description": "Old statement",
                "assertions": ["Old assertion"],
            },
            {
                "id": opik_suite.test_case_uuid("removed"),
                "data": {"task_id": "removed", "statement": "Removed"},
                "description": "Removed",
                "assertions": [],
            },
        ]
    )
    client = FakeClient(suite)
    monkeypatch.setattr(
        opik_suite,
        "_import_opik",
        lambda: SimpleNamespace(Opik=lambda: client),
    )
    monkeypatch.setenv("OPIK_PROJECT_NAME", "o11y-bench-test")

    result = opik_suite.sync_test_suite(specs_dir=tmp_path)

    assert result is suite
    assert [item["id"] for item in suite.updated] == [desired_id]
    assert suite.deleted == [opik_suite.test_case_uuid("removed")]
    assert suite.settings == [
        {
            "global_assertions": [],
            "global_execution_policy": {"runs_per_item": 3, "pass_threshold": 1},
        }
    ]
    assert client.create_kwargs["project_name"] == "o11y-bench-test"
    assert client.flushed is True


def test_execution_policy_rejects_invalid_threshold(monkeypatch):
    monkeypatch.setenv("OPIK_RUNS_PER_ITEM", "2")
    monkeypatch.setenv("OPIK_PASS_THRESHOLD", "3")

    with pytest.raises(ValueError, match="OPIK_PASS_THRESHOLD"):
        opik_suite._execution_policy()


def test_opik_url_strips_api_suffix(monkeypatch):
    monkeypatch.setenv("OPIK_URL_OVERRIDE", "http://localhost:5173/api/")

    assert opik_suite.opik_url() == "http://localhost:5173"
