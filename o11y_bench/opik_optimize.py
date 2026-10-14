"""Optimize the o11y-bench system prompt by evaluating candidates through Harbor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from .config import ROOT
from .opik_suite import sync_test_suite

_DEFAULT_JOBS_DIR = ROOT / "jobs" / "opik-optimization"
_RESULT_MARKER = "o11y_bench_optimization_result"
_MAX_REASON_LENGTH = 12_000


@dataclass(frozen=True, slots=True)
class OptimizationTask:
    task_id: str
    statement: str
    category: str
    tags: list[str]
    source_path: str

    def dataset_item(self, *, split: str) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "statement": self.statement,
            "category": self.category,
            "tags": self.tags,
            "source_path": self.source_path,
            "split": split,
        }


@dataclass(frozen=True, slots=True)
class TrialEvaluation:
    task_id: str
    candidate_hash: str
    score: float
    rewards: list[float]
    final_response: str
    feedback: list[str]
    job_dir: str

    def as_output(self) -> str:
        return json.dumps({_RESULT_MARKER: asdict(self)}, ensure_ascii=False)


class HarborOptimizationRunner:
    """Run one prompt candidate against one task using the normal benchmark path."""

    def __init__(
        self,
        *,
        root: Path,
        jobs_dir: Path,
        run_name: str,
        optimization_id: str,
        model: str,
        reasoning_effort: str,
        n_attempts: int,
        score_mode: str,
        publish_traces: bool,
        detailed_feedback: bool,
        scenario_time: str,
    ) -> None:
        self.root = root.resolve()
        self.jobs_dir = jobs_dir.resolve() / run_name
        self.run_name = run_name
        self.optimization_id = optimization_id
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.n_attempts = n_attempts
        self.score_mode = score_mode
        self.publish_traces = publish_traces
        self.detailed_feedback = detailed_feedback
        self.scenario_time = scenario_time

    def invoke(
        self,
        system_prompt: str,
        dataset_item: dict[str, Any],
        *,
        candidate_template: str | None = None,
    ) -> str:
        task_id = str(dataset_item.get("task_id") or "")
        if not task_id or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", task_id):
            raise ValueError(f"invalid optimization task_id: {task_id!r}")
        if not (self.root / "tasks" / task_id).is_dir():
            raise ValueError(f"generated Harbor task does not exist: {task_id}")

        candidate_source = candidate_template or system_prompt
        candidate_hash = hashlib.sha256(candidate_source.encode()).hexdigest()
        candidate_dir = self.jobs_dir / "candidates" / candidate_hash
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_stable_file(candidate_dir / "system_prompt.template.txt", candidate_source)
        prompt_path = candidate_dir / f"{task_id}.system_prompt.txt"
        _write_stable_file(prompt_path, system_prompt)

        job_name = f"{task_id}-opt-{candidate_hash[:16]}-k{self.n_attempts}"
        job_dir = self.jobs_dir / job_name
        command = [
            "uv",
            "run",
            "--extra",
            "opik",
            "python",
            "-m",
            "o11y_bench",
            "job",
            "--model",
            self.model,
            "--reasoning-effort",
            self.reasoning_effort,
            "--jobs-dir",
            str(self.jobs_dir),
            "--job-name",
            job_name,
            "--path",
            str(self.root / "tasks"),
            "--task-name",
            task_id,
            "--n-attempts",
            str(self.n_attempts),
            "--n-concurrent",
            "1",
            "--quiet",
            "--skip-preflight",
        ]
        if self.publish_traces:
            command.append("--opik-publish")
        command.extend(["--ak", f"system_prompt_path={prompt_path}"])

        env = os.environ.copy()
        env["O11Y_SCENARIO_TIME_ISO"] = self.scenario_time
        env["OPIK_OPTIMIZATION_ID"] = self.optimization_id
        env["OPIK_EXPERIMENT_NAME"] = f"{self.run_name}/{task_id}/{candidate_hash[:12]}"
        existing_tags = [tag for tag in env.get("OPIK_EXPERIMENT_TAGS", "").split(",") if tag]
        env["OPIK_EXPERIMENT_TAGS"] = ",".join(
            _dedupe([*existing_tags, "optimization", self.run_name, candidate_hash[:12]])
        )

        completed = subprocess.run(
            command,
            cwd=self.root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        log_path = candidate_dir / f"{task_id}.harbor.log"
        log_path.write_text(
            f"$ {' '.join(command)}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Harbor candidate evaluation failed for {task_id} "
                f"with exit code {completed.returncode}; see {log_path}"
            )

        evaluation = evaluate_job(
            job_dir,
            task_id=task_id,
            candidate_hash=candidate_hash,
            score_mode=self.score_mode,
            detailed_feedback=self.detailed_feedback,
        )
        return evaluation.as_output()


def load_optimization_tasks(specs_dir: Path = ROOT / "tasks-spec") -> dict[str, OptimizationTask]:
    """Load the optimizer-visible portion of task specs without hidden grading data."""

    tasks: dict[str, OptimizationTask] = {}
    for path in sorted(specs_dir.rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"task spec must be a mapping: {path}")
        task_id = str(payload.get("id") or "")
        statement = str(payload.get("statement") or "")
        category = str(payload.get("category") or "")
        raw_tags = payload.get("tags") or []
        if not task_id or not statement or not category or not isinstance(raw_tags, list):
            raise ValueError(f"task spec is missing optimizer fields: {path}")
        if task_id in tasks:
            raise ValueError(f"duplicate task id {task_id!r}: {path}")
        try:
            source_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            source_path = path.relative_to(specs_dir).as_posix()
        tasks[task_id] = OptimizationTask(
            task_id=task_id,
            statement=statement,
            category=category,
            tags=[str(tag) for tag in raw_tags],
            source_path=source_path,
        )
    if not tasks:
        raise ValueError(f"no task specs found under {specs_dir}")
    return tasks


def select_tasks(
    available: dict[str, OptimizationTask], task_ids: list[str], *, split: str
) -> list[OptimizationTask]:
    if not task_ids:
        return []
    duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate {split} task ids: {', '.join(duplicates)}")
    missing = [task_id for task_id in task_ids if task_id not in available]
    if missing:
        raise ValueError(f"unknown {split} task ids: {', '.join(missing)}")
    return [available[task_id] for task_id in task_ids]


def sync_optimization_dataset(
    client: Any,
    *,
    name: str,
    project_name: str,
    split: str,
    tasks: list[OptimizationTask],
) -> Any:
    """Create an ordinary Opik Dataset for one immutable optimization split."""

    dataset = client.get_or_create_dataset(
        name=name,
        description=f"o11y-bench {split} tasks for prompt optimization",
        project_name=project_name,
    )
    desired = {task.task_id: task.dataset_item(split=split) for task in tasks}
    existing_items = dataset.get_items()
    existing = {
        str(item.get("task_id")): item
        for item in existing_items
        if isinstance(item.get("task_id"), str)
    }
    new_items = [item for task_id, item in desired.items() if task_id not in existing]
    changed_items = [
        {**item, "id": existing[task_id]["id"]}
        for task_id, item in desired.items()
        if task_id in existing and _dataset_fields(existing[task_id]) != item
    ]
    stale_ids = [
        str(item["id"])
        for task_id, item in existing.items()
        if task_id not in desired and item.get("id")
    ]
    if new_items:
        dataset.insert(new_items, num_threads=1, deduplication=False)
    if changed_items:
        dataset.update(changed_items, deduplication=False)
    if stale_ids:
        dataset.delete(stale_ids)
    return dataset


def evaluate_job(
    job_dir: Path,
    *,
    task_id: str,
    candidate_hash: str,
    score_mode: str,
    detailed_feedback: bool,
) -> TrialEvaluation:
    trials: list[tuple[float, Path, dict[str, Any]]] = []
    if job_dir.is_dir():
        for result_path in sorted(job_dir.glob("*/result.json")):
            try:
                result = json.loads(result_path.read_text())
            except OSError, json.JSONDecodeError:
                continue
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            if rewards.get("reward") is None:
                continue
            trials.append((float(rewards["reward"]), result_path.parent, result))
    if not trials:
        raise RuntimeError(f"no scored Harbor trials found in {job_dir}")

    rewards = [trial[0] for trial in trials]
    if score_mode == "mean":
        score = fmean(rewards)
    elif score_mode == "pass-at-k":
        score = 1.0 if any(reward >= 1.0 for reward in rewards) else 0.0
    else:
        raise ValueError(f"unsupported score mode: {score_mode}")

    _best_reward, best_trial_dir, _best_result = max(trials, key=lambda trial: trial[0])
    feedback = _trial_feedback(best_trial_dir, detailed=detailed_feedback)
    return TrialEvaluation(
        task_id=task_id,
        candidate_hash=candidate_hash,
        score=score,
        rewards=rewards,
        final_response=_final_response(best_trial_dir),
        feedback=feedback,
        job_dir=str(job_dir),
    )


def parse_evaluation_output(value: str) -> TrialEvaluation:
    try:
        payload = json.loads(value)
        data = payload[_RESULT_MARKER]
        return TrialEvaluation(
            task_id=str(data["task_id"]),
            candidate_hash=str(data["candidate_hash"]),
            score=float(data["score"]),
            rewards=[float(item) for item in data["rewards"]],
            final_response=str(data["final_response"]),
            feedback=[str(item) for item in data["feedback"]],
            job_dir=str(data["job_dir"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("agent output did not contain an o11y-bench evaluation result") from exc


def benchmark_reward_metric(dataset_item: dict[str, Any], llm_output: str) -> Any:
    """Return the authoritative benchmark reward to Opik's optimization engine."""

    from opik.evaluation.metrics.score_result import ScoreResult

    del dataset_item
    evaluation = parse_evaluation_output(llm_output)
    reason_parts = [
        f"task={evaluation.task_id}",
        f"attempt_rewards={evaluation.rewards}",
        *evaluation.feedback,
        f"final_response={evaluation.final_response}",
    ]
    reason = "\n".join(reason_parts)
    if len(reason) > _MAX_REASON_LENGTH:
        reason = reason[:_MAX_REASON_LENGTH] + "\n[feedback truncated]"
    return ScoreResult(
        name="o11y_bench_reward",
        value=evaluation.score,
        reason=reason,
    )


def make_optimizable_agent(optimizer: Any, runner: HarborOptimizationRunner, project: str) -> Any:
    class HarborOptimizableAgent(optimizer.OptimizableAgent):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(project_name=project)
            self.trace_metadata = {
                "source": "o11y-bench",
                "framework": "harbor",
                "optimization_id": runner.optimization_id,
                "optimization_run": runner.run_name,
            }

        def invoke_agent(
            self,
            prompts: dict[str, Any],
            dataset_item: dict[str, Any],
            allow_tool_use: bool = False,
            seed: int | None = None,
        ) -> str:
            del allow_tool_use, seed
            if len(prompts) != 1:
                raise ValueError("o11y-bench optimization supports one system prompt")
            prompt = next(iter(prompts.values()))
            system_prompt_template = _system_prompt_text(prompt)
            system_prompt = _rendered_system_prompt(prompt, dataset_item)
            return runner.invoke(
                system_prompt,
                dataset_item,
                candidate_template=system_prompt_template,
            )

    return HarborOptimizableAgent()


def _rendered_system_prompt(prompt: Any, dataset_item: dict[str, Any]) -> str:
    get_messages = getattr(prompt, "get_messages", None)
    if callable(get_messages):
        messages = get_messages(dataset_item)
        for message in messages:
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                return str(message["content"])
    return _system_prompt_text(prompt)


def _system_prompt_text(prompt: Any) -> str:
    system_prompt = getattr(prompt, "system", None)
    if isinstance(system_prompt, str) and system_prompt.strip():
        return system_prompt
    messages = getattr(prompt, "messages", None) or []
    for message in messages:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            return str(message["content"])
    raise ValueError("candidate does not contain a system prompt")


def _trial_feedback(trial_dir: Path, *, detailed: bool) -> list[str]:
    path = trial_dir / "verifier" / "grading_details.json"
    try:
        details = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return []
    if not isinstance(details, dict):
        return []
    feedback: list[str] = []
    for name, value in details.items():
        if str(name).startswith("explanation:"):
            if detailed and value:
                feedback.append(f"{name}={value}")
            continue
        if isinstance(value, int | float) and not isinstance(value, bool):
            feedback.append(f"{name}={value}")
    return feedback


def _final_response(trial_dir: Path) -> str:
    path = trial_dir / "agent" / "trajectory.json"
    try:
        trajectory = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return ""
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else []
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if (
            isinstance(step, dict)
            and step.get("source") == "agent"
            and isinstance(step.get("message"), str)
            and step["message"]
        ):
            return str(step["message"])
    return ""


def _dataset_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in ("task_id", "statement", "category", "tags", "source_path", "split")
    }


def _write_stable_file(path: Path, content: str) -> None:
    if path.exists():
        existing = path.read_text()
        if existing != content:
            raise RuntimeError(f"candidate hash collision at {path}")
        return
    path.write_text(content)


def _scenario_time() -> str:
    configured = os.getenv("O11Y_SCENARIO_TIME_ISO", "").strip()
    if configured:
        return configured
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.") or "optimization"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _import_optimizer() -> Any:
    try:
        import opik_optimizer
    except ImportError as exc:
        raise RuntimeError(
            "Opik optimization requires opik-optimizer==3.2.0. "
            "Run `mise run opik:optimization:setup`."
        ) from exc
    return opik_optimizer


def _result_prompt(result: Any) -> Any:
    prompt = result.prompt
    if isinstance(prompt, dict):
        if len(prompt) != 1:
            raise ValueError("optimizer returned multiple prompts")
        return next(iter(prompt.values()))
    return prompt


def _save_result(result: Any, output_dir: Path, *, arguments: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = _result_prompt(result)
    system_prompt = _system_prompt_text(prompt)
    (output_dir / "best_system_prompt.txt").write_text(system_prompt)
    payload = {
        "optimization_id": result.optimization_id,
        "optimizer": result.optimizer,
        "score": result.score,
        "initial_score": result.initial_score,
        "metric_name": result.metric_name,
        "dataset_id": result.dataset_id,
        "llm_calls": result.llm_calls,
        "llm_cost_total": result.llm_cost_total,
        "llm_token_usage_total": result.llm_token_usage_total,
        "prompt": prompt.to_dict(),
        "arguments": arguments,
    }
    (output_dir / "optimization_result.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )


def resolve_run_manifest(
    path: Path,
    requested: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Create a run manifest or validate immutable settings before resuming."""

    if not path.exists():
        if resume:
            raise ValueError(f"cannot resume; optimization manifest does not exist: {path}")
        return requested
    if not resume:
        raise ValueError(
            f"optimization run already exists: {path.parent}. "
            "Use a new --run-name or pass --resume."
        )
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid optimization manifest: {path}") from exc
    stable_keys = (
        "run_name",
        "model",
        "optimizer_model",
        "training_tasks",
        "validation_tasks",
        "scenario_time",
        "score_mode",
        "n_attempts",
        "reasoning_effort",
        "project_name",
        "dataset_prefix",
        "publish_traces",
        "detailed_feedback",
        "seed",
    )
    mismatches = [key for key in stable_keys if existing.get(key) != requested.get(key)]
    if mismatches:
        raise ValueError("resume settings differ from the saved run: " + ", ".join(mismatches))
    return {
        **existing,
        "max_trials": requested["max_trials"],
        "prompts_per_round": requested["prompts_per_round"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize the o11y-bench agent system prompt through real Harbor scenarios"
    )
    parser.add_argument("--model", required=True, help="model evaluated by o11y-bench")
    parser.add_argument(
        "--optimizer-model",
        help="model that proposes prompt changes (defaults to --model)",
    )
    parser.add_argument("--task-name", action="append", required=True, dest="task_names")
    parser.add_argument(
        "--validation-task-name",
        action="append",
        default=[],
        dest="validation_task_names",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing --run-name with matching immutable settings",
    )
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--prompts-per-round", type=int, default=1)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="off")
    parser.add_argument("--score-mode", choices=("mean", "pass-at-k"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-name", default=os.getenv("OPIK_PROJECT_NAME", "o11y-bench"))
    parser.add_argument("--dataset-prefix")
    parser.add_argument("--jobs-dir", type=Path, default=_DEFAULT_JOBS_DIR)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-publish-traces", action="store_true")
    parser.add_argument("--detailed-feedback", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_trials < 1 or args.prompts_per_round < 1 or args.n_attempts < 1:
        raise SystemExit("--max-trials, --prompts-per-round, and --n-attempts must be positive")

    run_name = _slug(args.run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    all_tasks = load_optimization_tasks()
    training_tasks = select_tasks(all_tasks, args.task_names, split="training")
    validation_tasks = select_tasks(all_tasks, args.validation_task_names, split="validation")
    overlap = sorted(
        {task.task_id for task in training_tasks} & {task.task_id for task in validation_tasks}
    )
    if overlap:
        raise SystemExit(f"training and validation tasks overlap: {', '.join(overlap)}")

    publish_traces = not args.no_publish_traces
    dataset_prefix = args.dataset_prefix or f"o11y-bench-opt-{run_name}"
    output_dir = args.jobs_dir.resolve() / run_name
    manifest_path = output_dir / "optimization_manifest.json"
    requested_scenario_time = _scenario_time()
    if args.resume and manifest_path.exists():
        try:
            saved_manifest = json.loads(manifest_path.read_text())
            requested_scenario_time = str(
                saved_manifest.get("scenario_time") or requested_scenario_time
            )
        except OSError, json.JSONDecodeError:
            pass
    requested_manifest = {
        "run_name": run_name,
        "optimization_id": str(uuid.uuid7()),
        "model": args.model,
        "optimizer_model": args.optimizer_model or args.model,
        "training_tasks": [task.task_id for task in training_tasks],
        "validation_tasks": [task.task_id for task in validation_tasks],
        "scenario_time": requested_scenario_time,
        "score_mode": args.score_mode,
        "n_attempts": args.n_attempts,
        "reasoning_effort": args.reasoning_effort,
        "project_name": args.project_name,
        "dataset_prefix": dataset_prefix,
        "max_trials": args.max_trials,
        "prompts_per_round": args.prompts_per_round,
        "publish_traces": publish_traces,
        "detailed_feedback": args.detailed_feedback,
        "seed": args.seed,
    }
    try:
        run_manifest = resolve_run_manifest(
            manifest_path,
            requested_manifest,
            resume=args.resume,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    optimization_id = str(run_manifest["optimization_id"])
    scenario_time = str(run_manifest["scenario_time"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")

    optimizer_module = _import_optimizer()
    import opik

    if not args.skip_preflight:
        subprocess.run(
            ["bash", str(ROOT / "scripts" / "harbor_preflight.sh")],
            cwd=ROOT,
            check=True,
        )

    if publish_traces:
        sync_test_suite(progress=lambda message: print(message, flush=True))

    client = opik.Opik()
    training_dataset = sync_optimization_dataset(
        client,
        name=f"{dataset_prefix}-train",
        project_name=args.project_name,
        split="training",
        tasks=training_tasks,
    )
    validation_dataset = None
    if validation_tasks:
        validation_dataset = sync_optimization_dataset(
            client,
            name=f"{dataset_prefix}-validation",
            project_name=args.project_name,
            split="validation",
            tasks=validation_tasks,
        )
    if not args.resume:
        client.create_optimization(
            dataset_name=training_dataset.name,
            objective_name="benchmark_reward_metric",
            name=f"o11y-bench-{run_name}",
            metadata={
                "source": "o11y-bench",
                "framework": "harbor",
                "model": args.model,
                "optimizer_model": args.optimizer_model or args.model,
            },
            optimization_id=optimization_id,
            project_name=args.project_name,
        )
    client.flush()

    runner = HarborOptimizationRunner(
        root=ROOT,
        jobs_dir=args.jobs_dir,
        run_name=run_name,
        optimization_id=optimization_id,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        n_attempts=args.n_attempts,
        score_mode=args.score_mode,
        publish_traces=publish_traces,
        detailed_feedback=args.detailed_feedback,
        scenario_time=scenario_time,
    )
    agent = make_optimizable_agent(optimizer_module, runner, args.project_name)
    prompt = optimizer_module.ChatPrompt(
        name="o11y-bench-system-prompt",
        system=(ROOT / "agents" / "system_prompt.txt").read_text(),
        user="{statement}",
        model=args.model,
    )
    optimizer = optimizer_module.MetaPromptOptimizer(
        model=args.optimizer_model or args.model,
        prompts_per_round=args.prompts_per_round,
        n_threads=1,
        seed=args.seed,
        name=f"o11y-bench-{run_name}",
        verbose=1,
    )
    result = optimizer.optimize_prompt(
        prompt=prompt,
        dataset=training_dataset,
        validation_dataset=validation_dataset,
        metric=benchmark_reward_metric,
        agent=agent,
        n_samples=len(training_tasks),
        max_trials=args.max_trials,
        project_name=args.project_name,
        optimization_id=optimization_id,
        auto_continue=args.resume,
        allow_tool_use=False,
        optimize_prompts="system",
        experiment_config={
            "source": "o11y-bench",
            "framework": "harbor",
            **run_manifest,
        },
    )
    _save_result(result, output_dir, arguments=run_manifest)
    result.display()
    print(f"Best system prompt: {output_dir / 'best_system_prompt.txt'}")
    print(f"Optimization result: {output_dir / 'optimization_result.json'}")


if __name__ == "__main__":
    main()
