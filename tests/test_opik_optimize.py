from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from o11y_bench import opik_optimize


class FakeDataset:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.inserted = []
        self.updated = []
        self.deleted = []

    def get_items(self):
        return list(self.items)

    def insert(self, items, *, num_threads, deduplication):
        assert num_threads == 1
        assert deduplication is False
        self.inserted.extend(items)

    def update(self, items, *, deduplication):
        assert deduplication is False
        self.updated.extend(items)

    def delete(self, item_ids):
        self.deleted.extend(item_ids)


class FakeClient:
    def __init__(self, dataset):
        self.dataset = dataset
        self.kwargs = None

    def get_or_create_dataset(self, **kwargs):
        self.kwargs = kwargs
        return self.dataset


def _task(task_id="task-a", statement="Investigate."):
    return opik_optimize.OptimizationTask(
        task_id=task_id,
        statement=statement,
        category="investigation",
        tags=["metrics"],
        source_path=f"tasks-spec/test/{task_id}.yaml",
    )


def _write_scored_trial(job_dir, reward=0.75):
    trial_dir = job_dir / "task-a__trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir()
    (trial_dir / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": reward}}})
    )
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "agent", "message": "intermediate"},
                    {"source": "agent", "message": "final response"},
                ]
            }
        )
    )
    (trial_dir / "verifier" / "grading_details.json").write_text(
        json.dumps(
            {
                "score": reward,
                "checks_passed": 1,
                "explanation:hidden": "sensitive expected answer",
            }
        )
    )


def test_load_optimization_tasks_excludes_hidden_grading_data(tmp_path):
    path = tmp_path / "category" / "task.yaml"
    path.parent.mkdir()
    path.write_text(
        """\
id: task-a
category: investigation
tags: [metrics]
statement: Investigate.
facts:
  - name: hidden
    value: secret
rubric:
  - criterion: Find it.
    weight: 1
"""
    )

    tasks = opik_optimize.load_optimization_tasks(tmp_path)

    assert tasks["task-a"].task_id == "task-a"
    assert tasks["task-a"].statement == "Investigate."
    assert tasks["task-a"].source_path == "category/task.yaml"
    assert set(tasks["task-a"].dataset_item(split="training")) == {
        "task_id",
        "statement",
        "category",
        "tags",
        "source_path",
        "split",
    }


def test_select_tasks_rejects_unknown_and_duplicate_ids():
    available = {"task-a": _task()}

    with pytest.raises(ValueError, match="unknown training task ids"):
        opik_optimize.select_tasks(available, ["missing"], split="training")
    with pytest.raises(ValueError, match="duplicate training task ids"):
        opik_optimize.select_tasks(available, ["task-a", "task-a"], split="training")


def test_sync_optimization_dataset_reconciles_items():
    dataset = FakeDataset(
        [
            {**_task(statement="Old").dataset_item(split="training"), "id": "existing"},
            {**_task("removed").dataset_item(split="training"), "id": "removed"},
        ]
    )
    client = FakeClient(dataset)

    result = opik_optimize.sync_optimization_dataset(
        client,
        name="opt-train",
        project_name="o11y-bench",
        split="training",
        tasks=[_task(), _task("new")],
    )

    assert result is dataset
    assert dataset.inserted[0]["task_id"] == "new"
    assert dataset.updated[0]["id"] == "existing"
    assert dataset.updated[0]["statement"] == "Investigate."
    assert dataset.deleted == ["removed"]
    assert client.kwargs["project_name"] == "o11y-bench"


def test_harbor_runner_executes_candidate_and_returns_authoritative_score(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "tasks" / "task-a").mkdir(parents=True)
    jobs_dir = root / "jobs" / "optimization"
    prompt = "Use evidence and explain your reasoning."
    candidate_hash = hashlib.sha256(prompt.encode()).hexdigest()
    job_name = f"task-a-opt-{candidate_hash[:16]}-k1"
    job_dir = jobs_dir / "run-a" / job_name
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        _write_scored_trial(job_dir)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(opik_optimize.subprocess, "run", fake_run)
    runner = opik_optimize.HarborOptimizationRunner(
        root=root,
        jobs_dir=jobs_dir,
        run_name="run-a",
        optimization_id="018cc251-f400-7000-8000-000000000001",
        model="openai/gpt-test",
        reasoning_effort="off",
        n_attempts=1,
        score_mode="mean",
        publish_traces=True,
        detailed_feedback=False,
        scenario_time="2026-01-01T00:00:00Z",
    )

    output = runner.invoke(prompt, {"task_id": "task-a", "statement": "Investigate."})
    evaluation = opik_optimize.parse_evaluation_output(output)

    assert evaluation.score == 0.75
    assert evaluation.final_response == "final response"
    assert evaluation.feedback == ["score=0.75", "checks_passed=1"]
    command, kwargs = calls[0]
    assert "--skip-preflight" in command
    assert "--opik-publish" in command
    prompt_path = jobs_dir / "run-a" / "candidates" / candidate_hash / "task-a.system_prompt.txt"
    assert f"system_prompt_path={prompt_path}" in command
    assert kwargs["env"]["OPIK_OPTIMIZATION_ID"] == runner.optimization_id
    assert kwargs["env"]["O11Y_SCENARIO_TIME_ISO"] == "2026-01-01T00:00:00Z"


def test_evaluate_job_supports_pass_at_k_and_detailed_feedback(tmp_path):
    job_dir = tmp_path / "job"
    _write_scored_trial(job_dir, reward=0.2)
    second = job_dir / "task-a__second"
    (second / "agent").mkdir(parents=True)
    (second / "verifier").mkdir()
    (second / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}})
    )
    (second / "agent" / "trajectory.json").write_text(
        json.dumps({"steps": [{"source": "agent", "message": "best"}]})
    )
    (second / "verifier" / "grading_details.json").write_text(
        json.dumps({"score": 1.0, "explanation:criterion": "grounded"})
    )

    result = opik_optimize.evaluate_job(
        job_dir,
        task_id="task-a",
        candidate_hash="hash",
        score_mode="pass-at-k",
        detailed_feedback=True,
    )

    assert result.score == 1.0
    assert result.final_response == "best"
    assert "explanation:criterion=grounded" in result.feedback


def test_benchmark_metric_accepts_optimizer_keyword_arguments():
    evaluation = opik_optimize.TrialEvaluation(
        task_id="task-a",
        candidate_hash="hash",
        score=0.75,
        rewards=[0.75],
        final_response="answer",
        feedback=["score=0.75"],
        job_dir="/tmp/job",
    )

    score = opik_optimize.benchmark_reward_metric(
        dataset_item={"task_id": "task-a"},
        llm_output=evaluation.as_output(),
    )

    assert score.name == "o11y_bench_reward"
    assert score.value == 0.75


def test_parse_evaluation_output_rejects_regular_model_text():
    with pytest.raises(ValueError, match="did not contain"):
        opik_optimize.parse_evaluation_output("ordinary final answer")


def test_make_optimizable_agent_passes_candidate_system_prompt():
    invocations = []

    class FakeBaseAgent:
        def __init__(self, *, project_name):
            self.project_name = project_name
            self.trace_metadata = None

    class FakeRunner:
        optimization_id = "optimization-id"
        run_name = "run-a"

        def invoke(self, prompt, dataset_item, *, candidate_template=None):
            invocations.append((prompt, dataset_item, candidate_template))
            return "result"

    module = SimpleNamespace(OptimizableAgent=FakeBaseAgent)
    runner = FakeRunner()
    agent = opik_optimize.make_optimizable_agent(module, runner, "project")

    prompt = SimpleNamespace(
        system="candidate {task_id}",
        messages=None,
        get_messages=lambda item: [{"role": "system", "content": f"candidate {item['task_id']}"}],
    )
    result = agent.invoke_agent(
        {"main": prompt},
        {"task_id": "task-a"},
    )

    assert result == "result"
    assert invocations == [("candidate task-a", {"task_id": "task-a"}, "candidate {task_id}")]
    assert agent.trace_metadata["optimization_run"] == "run-a"
