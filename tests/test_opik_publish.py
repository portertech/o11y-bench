from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

from o11y_bench import opik_publish, opik_suite


class FakeExperimentNotFoundError(Exception):
    pass


class FakeRestClient:
    def __init__(self):
        self.finished = []

    def finish_experiments(self, *, ids):
        self.finished.append(ids)


class FakeExperiment:
    def __init__(self, experiment_id="experiment-id"):
        self.id = experiment_id
        self.uploads = []
        self.experiments_rest_client = FakeRestClient()

    def batch_upload_items(self, items, **kwargs):
        self.uploads.append((items, kwargs))


class FakeSuite:
    name = "o11y-bench"
    project_name = "o11y-bench-test"

    def get_items(self):
        return [
            {
                "id": opik_suite.test_case_uuid("task-a"),
                "data": {
                    "task_id": "task-a",
                    "statement": "Investigate.",
                    "category": "investigation",
                },
                "assertions": ["The answer is grounded."],
            }
        ]

    def get_version_info(self):
        return SimpleNamespace(id="version-1")


class FakeClient:
    def __init__(self, experiment):
        self.experiment = experiment
        self.create_kwargs = None
        self.attachments = []
        self.flushed = 0

    def get_test_suite(self, name, *, project_name):
        assert name == "o11y-bench"
        return FakeSuite()

    def get_experiment_by_id(self, _experiment_id):
        raise FakeExperimentNotFoundError

    def create_experiment(self, **kwargs):
        self.create_kwargs = kwargs
        return self.experiment

    def queue_attachment_upload(self, **kwargs):
        self.attachments.append(kwargs)

    def flush(self):
        self.flushed += 1
        return True


class FakeBulkObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeOpik:
    def __init__(self, client):
        self.client = client
        self.exceptions = SimpleNamespace(ExperimentNotFound=FakeExperimentNotFoundError)
        self.ExperimentItemBulkTrace = FakeBulkObject
        self.ExperimentItemBulkSpan = FakeBulkObject
        self.ExperimentItemBulkRecord = FakeBulkObject
        self.Opik = lambda: self.client


def _write_trial(tmp_path):
    tasks_dir = tmp_path / "tasks"
    problem_dir = tasks_dir / "task-a" / "tests"
    problem_dir.mkdir(parents=True)
    (problem_dir / "problem.yaml").write_text(
        """\
id: task-a
category: investigation
statement: Investigate.
checks:
  - name: queried_metrics
    weight: 1
    type: grounding
    params:
      mode: tool_trace_id
rubric:
  - criterion: The answer is grounded.
    weight: 1
"""
    )

    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "task-a__abc"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-01-01T00:00:00Z",
                        "source": "user",
                        "message": "Investigate.",
                    },
                    {
                        "step_id": 2,
                        "timestamp": "2026-01-01T00:00:01Z",
                        "source": "agent",
                        "message": "Checking metrics.",
                        "reasoning_content": "Need evidence.",
                        "tool_calls": [
                            {
                                "tool_call_id": "call-1",
                                "function_name": "query_prometheus",
                                "arguments": {"query": "up"},
                            }
                        ],
                        "observation": {"results": [{"source_call_id": "call-1", "content": "1"}]},
                        "metrics": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "cached_tokens": 2,
                            "cost_usd": 0.01,
                        },
                    },
                    {
                        "step_id": 3,
                        "timestamp": "2026-01-01T00:00:02Z",
                        "source": "agent",
                        "message": "Final grounded answer.",
                    },
                ],
            }
        )
    )
    (trial_dir / "verifier" / "grading_details.json").write_text(
        json.dumps(
            {
                "score": 1.0,
                "checks_passed": 1,
                "rubric_passed": 1,
                "queried_metrics": 1.0,
                "The answer is grounded.": 1.0,
                "explanation:queried_metrics": "metric query found",
                "explanation:The answer is grounded.": "answer used evidence",
            }
        )
    )
    result = {
        "id": "result-1",
        "task_name": "task-a",
        "task_checksum": "checksum-1",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:03Z",
        "agent_info": {
            "name": "o11y-bench",
            "version": "1.0.0",
            "model_info": {"provider": "anthropic", "name": "claude-test"},
        },
        "agent_result": {
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 5,
            "cost_usd": 0.01,
            "metadata": {"reasoning_effort": "high"},
        },
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
    }
    (trial_dir / "result.json").write_text(json.dumps(result))
    return job_dir, tasks_dir, trial_dir


def _publisher(monkeypatch, job_dir, tasks_dir):
    experiment = FakeExperiment()
    client = FakeClient(experiment)
    fake_opik = FakeOpik(client)
    monkeypatch.setattr(opik_publish, "_import_opik", lambda: fake_opik)
    publisher = opik_publish.OpikLivePublisher(
        job_dir,
        tasks_dir,
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        planned_trial_count=3,
    )
    return publisher, experiment, client


def test_live_publisher_uploads_trace_spans_scores_and_state(monkeypatch, tmp_path):
    job_dir, tasks_dir, _trial_dir = _write_trial(tmp_path)
    publisher, experiment, client = _publisher(monkeypatch, job_dir, tasks_dir)

    publisher._connect()
    publisher.scan_once()

    assert client.create_kwargs["evaluation_method"] == "evaluation_suite"
    assert client.create_kwargs["dataset_version_id"] == "version-1"
    assert client.create_kwargs["experiment_config"]["planned_trial_count"] == 3
    assert len(experiment.uploads) == 1
    records, upload_kwargs = experiment.uploads[0]
    assert upload_kwargs == {"project_name": "o11y-bench-test", "num_threads": 1}
    record = records[0]
    assert record.dataset_item_id == opik_suite.test_case_uuid("task-a")
    assert uuid.UUID(record.trace.id).version == 7
    assert all(uuid.UUID(span.id).version == 7 for span in record.spans)
    assert record.trace.output == {"response": "Final grounded answer."}
    assert [span.type for span in record.spans] == ["general", "llm", "tool", "llm"]
    assert {score["name"] for score in record.feedback_scores} == {
        "final",
        "check.checks-passed",
        "rubric.rubric-passed",
        "check.queried-metrics",
        "rubric.the-answer-is-grounded",
    }

    state = json.loads((job_dir / ".opik-publish.json").read_text())
    assert state["exported_trials"] == ["task-a__abc"]
    assert state["score_count"] == 5


def test_publisher_reuses_state_without_duplicate_upload(monkeypatch, tmp_path):
    job_dir, tasks_dir, _trial_dir = _write_trial(tmp_path)
    first, first_experiment, _client = _publisher(monkeypatch, job_dir, tasks_dir)
    first._connect()
    first.scan_once()
    assert len(first_experiment.uploads) == 1

    second, second_experiment, _client = _publisher(monkeypatch, job_dir, tasks_dir)
    second._connect()
    second.scan_once()

    assert second._trial_count == 1
    assert second._score_count == 5
    assert second_experiment.uploads == []


def test_publisher_links_harbor_experiment_to_optimization(monkeypatch, tmp_path):
    job_dir, tasks_dir, _trial_dir = _write_trial(tmp_path)
    experiment = FakeExperiment()
    client = FakeClient(experiment)
    fake_opik = FakeOpik(client)
    monkeypatch.setattr(opik_publish, "_import_opik", lambda: fake_opik)
    optimization_id = "018cc251-f400-7000-8000-000000000001"
    publisher = opik_publish.OpikLivePublisher(
        job_dir,
        tasks_dir,
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=opik_publish.OpikPublishOptions(optimization_id=optimization_id),
    )

    publisher._connect()
    publisher.scan_once()

    assert client.create_kwargs["optimization_id"] == optimization_id
    record = experiment.uploads[0][0][0]
    assert record.trace.metadata["optimization_id"] == optimization_id


def test_finish_marks_successful_experiment_complete(monkeypatch, tmp_path):
    job_dir, tasks_dir, _trial_dir = _write_trial(tmp_path)
    publisher, experiment, client = _publisher(monkeypatch, job_dir, tasks_dir)
    publisher._connect()

    result = publisher.finish(succeeded=True)

    assert result.trial_count == 1
    assert experiment.experiments_rest_client.finished == [[publisher.experiment_id]]
    assert client.flushed == 1


def test_upload_artifacts_is_opt_in(monkeypatch, tmp_path):
    job_dir, tasks_dir, trial_dir = _write_trial(tmp_path)
    experiment = FakeExperiment()
    client = FakeClient(experiment)
    fake_opik = FakeOpik(client)
    monkeypatch.setattr(opik_publish, "_import_opik", lambda: fake_opik)
    publisher = opik_publish.OpikLivePublisher(
        job_dir,
        tasks_dir,
        model="anthropic/claude-test",
        agent_name="o11y-bench",
        agent_version="test",
        options=opik_publish.OpikPublishOptions(upload_artifacts=True),
    )
    publisher._connect()
    publisher.scan_once()

    assert {item["file_path"] for item in client.attachments} == {
        str(trial_dir / "agent" / "trajectory.json"),
        str(trial_dir / "verifier" / "grading_details.json"),
        str(trial_dir / "result.json"),
    }


def test_publish_options_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("OPIK_UPLOAD_ARTIFACTS", "sometimes")

    with pytest.raises(ValueError, match="OPIK_UPLOAD_ARTIFACTS"):
        opik_publish.OpikPublishOptions.from_env()


def test_explicit_experiment_id_must_be_uuid7(tmp_path):
    with pytest.raises(ValueError, match="OPIK_EXPERIMENT_ID"):
        opik_publish.OpikLivePublisher(
            tmp_path / "job",
            tmp_path / "tasks",
            model="anthropic/test",
            agent_name="agent",
            agent_version="test",
            options=opik_publish.OpikPublishOptions(experiment_id="not-a-uuid"),
        )
