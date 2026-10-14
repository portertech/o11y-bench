"""Stream completed Harbor trials into an Opik Test Suite experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from grading.transcript_parser import parse_transcript
from reporting.report_data import (
    agent_result_metrics,
    load_trials,
    parse_datetime,
    trial_reasoning_effort,
)

from .opik_suite import DEFAULT_TEST_SUITE_NAME, opik_url

_STATE_FILE = ".opik-publish.json"
_EXPERIMENT_NAMESPACE = uuid.UUID("4982e908-b247-5507-a58a-a87cc8dff059")
_TRACE_NAMESPACE = uuid.UUID("d29e7995-b2e1-59a5-a6e6-af2da389311f")
_SPAN_NAMESPACE = uuid.UUID("30b595ee-bd07-56f3-b38a-cad5fc326c30")
_MAX_TEXT_LENGTH = 32_000
_MAX_COLLECTION_LENGTH = 100


@dataclass(slots=True)
class OpikPublishOptions:
    """Environment-driven settings for one Opik experiment."""

    experiment_id: str = ""
    suite_name: str = DEFAULT_TEST_SUITE_NAME
    name: str = ""
    tags: list[str] = field(default_factory=lambda: ["o11y-bench"])
    upload_artifacts: bool = False

    @classmethod
    def from_env(cls) -> OpikPublishOptions:
        raw_tags = os.getenv("OPIK_EXPERIMENT_TAGS", "")
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
        return cls(
            experiment_id=os.getenv("OPIK_EXPERIMENT_ID", "").strip(),
            suite_name=os.getenv("OPIK_TEST_SUITE_NAME", DEFAULT_TEST_SUITE_NAME).strip()
            or DEFAULT_TEST_SUITE_NAME,
            name=os.getenv("OPIK_EXPERIMENT_NAME", "").strip(),
            tags=["o11y-bench", *tags],
            upload_artifacts=_env_bool("OPIK_UPLOAD_ARTIFACTS", default=False),
        )


@dataclass(frozen=True, slots=True)
class OpikPublishResult:
    experiment_id: str
    experiment_name: str
    trial_count: int
    score_count: int
    url: str


class OpikLivePublisher:
    """Publish each scored Harbor trial when its result artifact appears."""

    def __init__(
        self,
        job_dir: Path,
        tasks_dir: Path,
        *,
        model: str,
        agent_name: str,
        agent_version: str,
        options: OpikPublishOptions | None = None,
        planned_trial_count: int | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.job_dir = job_dir.resolve()
        self.tasks_dir = tasks_dir.resolve()
        self.model = model
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.options = options or OpikPublishOptions.from_env()
        self.planned_trial_count = planned_trial_count
        self.poll_interval_seconds = poll_interval_seconds
        self.experiment_id = self.options.experiment_id or _default_experiment_id(
            self.options.suite_name, self.job_dir.name
        )
        _validate_uuid(self.experiment_id, "OPIK_EXPERIMENT_ID")
        self.experiment_name = self.options.name or f"o11y-bench {self.job_dir.name}"
        self._opik: Any | None = None
        self._client: Any | None = None
        self._suite: Any | None = None
        self._experiment: Any | None = None
        self._suite_cases: dict[str, dict[str, Any]] = {}
        self._project_name: str | None = None
        self._exported: set[str] = set()
        self._trial_count = 0
        self._score_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._scan_lock = threading.Lock()
        self._error: Exception | None = None
        self.last_result: OpikPublishResult | None = None
        self._load_state()

    @property
    def url(self) -> str:
        return opik_url()

    def start(self) -> None:
        self._connect()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="opik-live-publish",
            daemon=True,
        )
        self._thread.start()

    def finish(self, *, succeeded: bool, error: str = "") -> OpikPublishResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_interval_seconds * 3))

        publish_error = self._error
        try:
            self.scan_once()
            if succeeded and self._experiment is not None:
                self._experiment.experiments_rest_client.finish_experiments(
                    ids=[self.experiment_id]
                )
            if self._client is not None:
                self._client.flush()
        except Exception as exc:
            publish_error = publish_error or exc

        self.last_result = OpikPublishResult(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            trial_count=self._trial_count,
            score_count=self._score_count,
            url=self.url,
        )
        if publish_error is not None:
            raise RuntimeError(f"Opik live publishing failed: {publish_error}") from publish_error
        return self.last_result

    def publish_existing(self) -> OpikPublishResult:
        self._connect()
        self.scan_once()
        return self.finish(succeeded=True)

    def scan_once(self) -> None:
        if self._experiment is None or not self.job_dir.exists():
            return
        with self._scan_lock:
            for result in load_trials(self.job_dir):
                result_path = Path(str(result["__result_path"]))
                trial_dir = result_path.parent
                harbor_trial_id = trial_dir.name
                rewards = (result.get("verifier_result") or {}).get("rewards") or {}
                if harbor_trial_id in self._exported or rewards.get("reward") is None:
                    continue
                self._publish_trial(result, trial_dir, rewards)
                self._exported.add(harbor_trial_id)
                self._trial_count += 1
                self._save_state()
                if self.options.upload_artifacts:
                    self._upload_artifacts(
                        trial_dir, _trace_id(self.experiment_id, harbor_trial_id)
                    )

    def _connect(self) -> None:
        if self._experiment is not None:
            return
        opik = _import_opik()
        client = opik.Opik()
        project_name = os.getenv("OPIK_PROJECT_NAME", "").strip() or None
        try:
            suite = client.get_test_suite(self.options.suite_name, project_name=project_name)
        except Exception as exc:
            raise RuntimeError(
                f"Opik Test Suite {self.options.suite_name!r} was not found. "
                "Run `mise run opik:sync-suite` first."
            ) from exc

        suite_cases: dict[str, dict[str, Any]] = {}
        for item in suite.get_items():
            data = item.get("data") or {}
            task_id = data.get("task_id")
            if isinstance(task_id, str) and task_id:
                suite_cases[task_id] = item

        try:
            experiment = client.get_experiment_by_id(self.experiment_id)
        except opik.exceptions.ExperimentNotFound:
            version_info = suite.get_version_info()
            experiment = client.create_experiment(
                dataset_name=suite.name,
                name=self.experiment_name,
                experiment_id=self.experiment_id,
                experiment_config={
                    "source": "o11y-bench",
                    "job_name": self.job_dir.name,
                    "framework": "harbor",
                    "transcript_format": "ATIF",
                    "agent_name": self.agent_name,
                    "agent_version": self.agent_version,
                    "model": self.model,
                    "planned_trial_count": self.planned_trial_count,
                },
                evaluation_method="evaluation_suite",
                tags=_dedupe([*self.options.tags, self.agent_name, self.model]),
                dataset_version_id=getattr(version_info, "id", None),
                project_name=suite.project_name,
            )

        self._opik = opik
        self._client = client
        self._suite = suite
        self._experiment = experiment
        self._suite_cases = suite_cases
        self._project_name = suite.project_name or project_name

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            try:
                self.scan_once()
            except Exception as exc:
                self._error = exc
                self._stop.set()

    def _publish_trial(
        self,
        result: dict[str, Any],
        trial_dir: Path,
        rewards: dict[str, Any],
    ) -> None:
        assert self._opik is not None
        assert self._experiment is not None
        task_id = str(result.get("task_name") or trial_dir.name.split("__", 1)[0])
        case = self._suite_cases.get(task_id)
        if case is None:
            raise ValueError(f"Harbor result references task {task_id!r}, absent from Opik suite")

        harbor_trial_id = trial_dir.name
        trace_id = _trace_id(self.experiment_id, harbor_trial_id)
        trace, spans = _trace_and_spans(
            self._opik,
            result,
            trial_dir,
            case,
            trace_id=trace_id,
            project_name=self._project_name,
        )
        feedback_scores = _feedback_scores(result, trial_dir, rewards, self.tasks_dir)
        record = self._opik.ExperimentItemBulkRecord(
            dataset_item_id=str(case["id"]),
            trace=trace,
            spans=spans,
            feedback_scores=feedback_scores,
        )
        self._experiment.batch_upload_items(
            [record],
            project_name=self._project_name,
            num_threads=1,
        )
        self._score_count += len(feedback_scores)

    def _upload_artifacts(self, trial_dir: Path, trace_id: str) -> None:
        assert self._client is not None
        if not self._project_name:
            return
        for path in _artifact_paths(trial_dir):
            self._client.queue_attachment_upload(
                entity_type="trace",
                entity_id=trace_id,
                project_name=self._project_name,
                file_path=str(path),
            )
        if not self._client.flush():
            raise RuntimeError(f"Opik did not finish uploading artifacts for {trial_dir.name}")

    def _load_state(self) -> None:
        path = self.job_dir / _STATE_FILE
        try:
            payload = json.loads(path.read_text())
        except OSError, json.JSONDecodeError:
            return
        if payload.get("experiment_id") != self.experiment_id:
            return
        self._exported = {str(item) for item in payload.get("exported_trials", [])}
        self._trial_count = len(self._exported)
        self._score_count = int(payload.get("score_count") or 0)

    def _save_state(self) -> None:
        if not self.job_dir.exists():
            return
        payload = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "exported_trials": sorted(self._exported),
            "score_count": self._score_count,
        }
        (self.job_dir / _STATE_FILE).write_text(json.dumps(payload, indent=2) + "\n")


def _trace_and_spans(
    opik: Any,
    result: dict[str, Any],
    trial_dir: Path,
    case: dict[str, Any],
    *,
    trace_id: str,
    project_name: str | None,
) -> tuple[Any, list[Any]]:
    started_at = _result_datetime(result, "started_at")
    finished_at = _result_datetime(result, "finished_at", fallback=started_at)
    task_id = str(result.get("task_name") or trial_dir.name.split("__", 1)[0])
    case_data = case.get("data") or {}
    provider, model_name = _result_model_ref(result)
    cost, input_tokens, cache_tokens, output_tokens = agent_result_metrics(result)
    final_output = _final_output(trial_dir)

    trace = opik.ExperimentItemBulkTrace(
        id=trace_id,
        name=f"o11y-bench/{task_id}",
        project_name=project_name,
        start_time=started_at,
        end_time=max(started_at, finished_at),
        input={"task_id": task_id, "statement": str(case_data.get("statement") or "")},
        output={"response": _bounded(final_output)},
        metadata={
            "source": "o11y-bench",
            "framework": "harbor",
            "harbor_trial_id": trial_dir.name,
            "harbor_result_id": result.get("id"),
            "agent_name": (result.get("agent_info") or {}).get("name"),
            "agent_version": (result.get("agent_info") or {}).get("version"),
            "model_provider": provider,
            "model_name": model_name,
            "reasoning_effort": trial_reasoning_effort(result),
            "task_checksum": result.get("task_checksum"),
            "usage": {
                "input_tokens": input_tokens,
                "cached_input_tokens": cache_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            },
        },
        tags=_dedupe(
            [
                "o11y-bench",
                str(case_data.get("category") or ""),
                provider,
                model_name,
            ]
        ),
        error_info=_error_info(result.get("exception_info")),
    )
    spans = _trajectory_spans(
        opik,
        trial_dir,
        trace_id=trace_id,
        provider=provider,
        model_name=model_name,
        default_start=started_at,
        default_end=finished_at,
    )
    return trace, spans


def _trajectory_spans(
    opik: Any,
    trial_dir: Path,
    *,
    trace_id: str,
    provider: str,
    model_name: str,
    default_start: datetime,
    default_end: datetime,
) -> list[Any]:
    trajectory = _load_trajectory(trial_dir)
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if not isinstance(steps, list):
        return []

    timestamps = [
        parse_datetime(step.get("timestamp")) if isinstance(step, dict) else None for step in steps
    ]
    spans: list[Any] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            continue
        source = str(raw_step.get("source") or "unknown")
        start_time = timestamps[index] or default_start
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else None
        end_time = next_time or max(start_time + timedelta(milliseconds=1), default_end)
        if end_time < start_time:
            end_time = start_time + timedelta(milliseconds=1)
        step_id = str(raw_step.get("step_id") or index + 1)
        span_id = _span_id(trace_id, f"step:{step_id}:{index}")
        metrics = raw_step.get("metrics") or {}
        tool_calls = raw_step.get("tool_calls") or []
        observation = raw_step.get("observation") or {}
        observation_results = observation.get("results") if isinstance(observation, dict) else []
        if not isinstance(observation_results, list):
            observation_results = []

        usage = _step_usage(metrics)
        spans.append(
            opik.ExperimentItemBulkSpan(
                id=span_id,
                name=f"step_{step_id}_{source}",
                type="llm" if source == "agent" else "general",
                start_time=start_time,
                end_time=end_time,
                input={
                    "message": _bounded(raw_step.get("message")),
                    "tool_calls": _bounded(tool_calls),
                },
                output=(
                    {"observation": _bounded(observation_results)}
                    if observation_results and not tool_calls
                    else None
                ),
                metadata={
                    "source": source,
                    "reasoning": _bounded(raw_step.get("reasoning_content")),
                    "cached_tokens": _number(metrics.get("cached_tokens")),
                },
                model=model_name if source == "agent" else None,
                provider=provider if source == "agent" else None,
                tags=["harbor", source],
                usage=usage,
                total_estimated_cost=_optional_float(metrics.get("cost_usd")),
            )
        )

        results_by_call = {
            str(item.get("source_call_id") or item.get("tool_call_id") or ""): item
            for item in observation_results
            if isinstance(item, dict)
        }
        for tool_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            call_id = str(tool_call.get("tool_call_id") or tool_call.get("id") or tool_index)
            function_name = str(tool_call.get("function_name") or tool_call.get("name") or "tool")
            tool_result = results_by_call.get(call_id) or {}
            spans.append(
                opik.ExperimentItemBulkSpan(
                    id=_span_id(trace_id, f"step:{step_id}:tool:{call_id}:{tool_index}"),
                    parent_span_id=span_id,
                    name=function_name,
                    type="tool",
                    start_time=start_time,
                    end_time=end_time,
                    input={"arguments": _bounded(tool_call.get("arguments") or {})},
                    output={"content": _bounded(tool_result.get("content"))},
                    metadata={"tool_call_id": call_id, "source": "harbor"},
                    tags=["harbor", "tool"],
                )
            )
    return spans


def _feedback_scores(
    result: dict[str, Any],
    trial_dir: Path,
    rewards: dict[str, Any],
    tasks_dir: Path,
) -> list[dict[str, Any]]:
    details = _grading_details(trial_dir)
    check_names, rubric_names = _evaluator_names(tasks_dir, str(result.get("task_name") or ""))
    final_value = float(rewards["reward"])
    scores: list[dict[str, Any]] = [
        {
            "name": "final",
            "value": final_value,
            "category_name": "o11y-bench",
            "reason": _bounded_text(_overall_explanation(details)),
        }
    ]
    values: dict[str, Any] = {
        key: value for key, value in details.items() if not str(key).startswith("explanation:")
    }
    values.update({key: value for key, value in rewards.items() if key != "reward"})
    values.pop("score", None)

    for name, raw_value in values.items():
        try:
            value = float(raw_value)
        except TypeError, ValueError:
            continue
        if name in check_names or name in {"checks_passed", "validators_passed"}:
            prefix = "check"
        elif name in rubric_names or name == "rubric_passed":
            prefix = "rubric"
        else:
            prefix = "score"
        scores.append(
            {
                "name": f"{prefix}.{_slug(name)}",
                "value": value,
                "category_name": prefix,
                "reason": _bounded_text(str(details.get(f"explanation:{name}") or "")),
            }
        )
    return scores


def _evaluator_names(tasks_dir: Path, task_name: str) -> tuple[set[str], set[str]]:
    path = tasks_dir / task_name / "tests" / "problem.yaml"
    try:
        payload = yaml.safe_load(path.read_text())
    except OSError, yaml.YAMLError:
        return set(), set()
    if not isinstance(payload, dict):
        return set(), set()
    checks = {
        str(item.get("name"))
        for item in payload.get("checks", [])
        if isinstance(item, dict) and item.get("name")
    }
    rubrics = {
        str(item.get("criterion"))
        for item in payload.get("rubric", [])
        if isinstance(item, dict) and item.get("criterion")
    }
    return checks, rubrics


def _load_trajectory(trial_dir: Path) -> dict[str, Any]:
    candidates = [trial_dir / "agent" / "trajectory.json"]
    agent_dir = trial_dir / "agent"
    if agent_dir.exists():
        candidates.extend(sorted(agent_dir.rglob("trajectory.json")))
    for path in dict.fromkeys(candidates):
        try:
            payload = json.loads(path.read_text())
        except OSError, json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
            return payload
    return {}


def _final_output(trial_dir: Path) -> str:
    transcript = parse_transcript(trial_dir / "agent")
    for message in reversed(transcript.messages):
        if message.role == "assistant" and message.content:
            return message.content
    return ""


def _grading_details(trial_dir: Path) -> dict[str, Any]:
    path = trial_dir / "verifier" / "grading_details.json"
    try:
        payload = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _overall_explanation(details: dict[str, Any]) -> str:
    return "\n".join(
        str(value)
        for key, value in details.items()
        if str(key).startswith("explanation:") and value
    )


def _artifact_paths(trial_dir: Path) -> list[Path]:
    candidates = [
        trial_dir / "agent" / "trajectory.json",
        trial_dir / "verifier" / "grading_details.json",
        trial_dir / "result.json",
    ]
    return [path for path in candidates if path.exists()]


def _result_datetime(
    result: dict[str, Any], key: str, *, fallback: datetime | None = None
) -> datetime:
    parsed = parse_datetime(result.get(key))
    if parsed is not None:
        return parsed
    return fallback or datetime.now(UTC)


def _result_model_ref(result: dict[str, Any]) -> tuple[str, str]:
    model_info = (result.get("agent_info") or {}).get("model_info") or {}
    provider = str(model_info.get("provider") or "")
    name = str(model_info.get("name") or "")
    if "/" in name:
        embedded_provider, name = name.split("/", 1)
        provider = provider or embedded_provider
    return provider, name


def _error_info(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    exception_type = str(value.get("exception_type") or "HarborTrialError")
    message = str(value.get("exception_message") or value.get("message") or "")
    traceback = str(value.get("traceback") or value.get("exception_traceback") or "")
    return {"exception_type": exception_type, "message": message, "traceback": traceback}


def _step_usage(metrics: Any) -> dict[str, int] | None:
    if not isinstance(metrics, dict):
        return None
    prompt = int(_number(metrics.get("prompt_tokens")))
    completion = int(_number(metrics.get("completion_tokens")))
    if prompt == 0 and completion == 0:
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except TypeError, ValueError:
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _bounded(value: Any) -> Any:
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, list):
        values = [_bounded(item) for item in value[:_MAX_COLLECTION_LENGTH]]
        if len(value) > _MAX_COLLECTION_LENGTH:
            values.append(f"[truncated {len(value) - _MAX_COLLECTION_LENGTH} items]")
        return values
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            str(key): _bounded(item_value) for key, item_value in items[:_MAX_COLLECTION_LENGTH]
        }
        if len(items) > _MAX_COLLECTION_LENGTH:
            bounded["_truncated"] = len(items) - _MAX_COLLECTION_LENGTH
        return bounded
    if value is None or isinstance(value, bool | int | float):
        return value
    return _bounded_text(str(value))


def _bounded_text(value: str) -> str:
    if len(value) <= _MAX_TEXT_LENGTH:
        return value
    removed = len(value) - _MAX_TEXT_LENGTH
    return f"{value[:_MAX_TEXT_LENGTH]}\n[truncated {removed} characters]"


def _default_experiment_id(suite_name: str, job_name: str) -> str:
    return str(uuid.uuid5(_EXPERIMENT_NAMESPACE, f"{suite_name}\x1f{job_name}"))


def _trace_id(experiment_id: str, harbor_trial_id: str) -> str:
    return str(uuid.uuid5(_TRACE_NAMESPACE, f"{experiment_id}\x1f{harbor_trial_id}"))


def _span_id(trace_id: str, key: str) -> str:
    return str(uuid.uuid5(_SPAN_NAMESPACE, f"{trace_id}\x1f{key}"))


def _validate_uuid(value: str, name: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}") from exc


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "score"
    if len(normalized) <= 80:
        return normalized
    digest = hashlib.sha1(value.encode()).hexdigest()[:10]
    return f"{normalized[:69]}-{digest}"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _import_opik() -> Any:
    try:
        import opik
    except ImportError as exc:
        raise RuntimeError(
            "Opik publishing requires opik==2.2.100. "
            "Run `mise run opik:setup` to install the optional integration."
        ) from exc
    return opik
