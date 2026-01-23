"""
Data types for ADK-RLM.

These types are designed to match the original RLM implementation for
compatibility with the visualizer.
"""

from dataclasses import dataclass
from dataclasses import field
from types import ModuleType
from typing import Any


def _serialize_value(value: Any) -> Any:
  """Convert a value to a JSON-serializable representation."""
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, ModuleType):
    return f"<module '{value.__name__}'>"
  if isinstance(value, (list, tuple)):
    return [_serialize_value(v) for v in value]
  if isinstance(value, dict):
    return {str(k): _serialize_value(v) for k, v in value.items()}
  if callable(value):
    return (
        f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    )
  try:
    return repr(value)
  except Exception:
    return f"<{type(value).__name__}>"


@dataclass
class ModelUsageSummary:
  """Usage summary for a single model."""

  total_calls: int
  total_input_tokens: int
  total_output_tokens: int

  def to_dict(self) -> dict[str, Any]:
    return {
        "total_calls": self.total_calls,
        "total_input_tokens": self.total_input_tokens,
        "total_output_tokens": self.total_output_tokens,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "ModelUsageSummary":
    return cls(
        total_calls=data.get("total_calls", 0),
        total_input_tokens=data.get("total_input_tokens", 0),
        total_output_tokens=data.get("total_output_tokens", 0),
    )


@dataclass
class UsageSummary:
  """Aggregated usage summary across all models."""

  model_usage_summaries: dict[str, ModelUsageSummary] = field(
      default_factory=dict
  )

  def to_dict(self) -> dict[str, Any]:
    return {
        "model_usage_summaries": {
            model: usage.to_dict()
            for model, usage in self.model_usage_summaries.items()
        },
    }

  @classmethod
  def from_dict(cls, data: dict) -> "UsageSummary":
    return cls(
        model_usage_summaries={
            model: ModelUsageSummary.from_dict(usage)
            for model, usage in data.get("model_usage_summaries", {}).items()
        },
    )

  @property
  def total_calls(self) -> int:
    return sum(m.total_calls for m in self.model_usage_summaries.values())

  @property
  def total_input_tokens(self) -> int:
    return sum(
        m.total_input_tokens for m in self.model_usage_summaries.values()
    )

  @property
  def total_output_tokens(self) -> int:
    return sum(
        m.total_output_tokens for m in self.model_usage_summaries.values()
    )


@dataclass
class RLMChatCompletion:
  """Record of a single LLM call made from within the environment."""

  root_model: str
  prompt: str | dict[str, Any]
  response: str
  usage_summary: UsageSummary
  execution_time: float

  def to_dict(self) -> dict[str, Any]:
    return {
        "root_model": self.root_model,
        "prompt": self.prompt,
        "response": self.response,
        "usage_summary": self.usage_summary.to_dict(),
        "execution_time": self.execution_time,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "RLMChatCompletion":
    return cls(
        root_model=data.get("root_model", ""),
        prompt=data.get("prompt", ""),
        response=data.get("response", ""),
        usage_summary=UsageSummary.from_dict(data.get("usage_summary", {})),
        execution_time=data.get("execution_time", 0.0),
    )


@dataclass
class REPLResult:
  """Result from executing code in the REPL environment."""

  stdout: str
  stderr: str
  locals: dict[str, Any]
  execution_time: float
  rlm_calls: list[RLMChatCompletion] = field(default_factory=list)

  def __str__(self) -> str:
    return (
        f"REPLResult(stdout={self.stdout!r}, stderr={self.stderr!r},"
        f" locals={list(self.locals.keys())},"
        f" execution_time={self.execution_time:.3f}s,"
        f" rlm_calls={len(self.rlm_calls)})"
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "stdout": self.stdout,
        "stderr": self.stderr,
        "locals": {k: _serialize_value(v) for k, v in self.locals.items()},
        "execution_time": self.execution_time,
        "rlm_calls": [call.to_dict() for call in self.rlm_calls],
    }

  @classmethod
  def from_dict(cls, data: dict) -> "REPLResult":
    return cls(
        stdout=data.get("stdout", ""),
        stderr=data.get("stderr", ""),
        locals=data.get("locals", {}),
        execution_time=data.get("execution_time", 0.0),
        rlm_calls=[
            RLMChatCompletion.from_dict(c) for c in data.get("rlm_calls", [])
        ],
    )


@dataclass
class CodeBlock:
  """A code block extracted from an LLM response with its execution result."""

  code: str
  result: REPLResult

  def to_dict(self) -> dict[str, Any]:
    return {
        "code": self.code,
        "result": self.result.to_dict(),
    }

  @classmethod
  def from_dict(cls, data: dict) -> "CodeBlock":
    return cls(
        code=data.get("code", ""),
        result=REPLResult.from_dict(data.get("result", {})),
    )


@dataclass
class RLMIteration:
  """A single iteration of the RLM loop."""

  prompt: str | dict[str, Any]
  response: str
  code_blocks: list[CodeBlock]
  final_answer: str | None = None
  iteration_time: float | None = None

  def to_dict(self) -> dict[str, Any]:
    return {
        "prompt": self.prompt,
        "response": self.response,
        "code_blocks": [cb.to_dict() for cb in self.code_blocks],
        "final_answer": self.final_answer,
        "iteration_time": self.iteration_time,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "RLMIteration":
    return cls(
        prompt=data.get("prompt", ""),
        response=data.get("response", ""),
        code_blocks=[
            CodeBlock.from_dict(cb) for cb in data.get("code_blocks", [])
        ],
        final_answer=data.get("final_answer"),
        iteration_time=data.get("iteration_time"),
    )


@dataclass
class RLMMetadata:
  """Metadata about the RLM configuration."""

  root_model: str
  max_depth: int
  max_iterations: int
  backend: str
  backend_kwargs: dict[str, Any]
  environment_type: str
  environment_kwargs: dict[str, Any]
  other_backends: list[str] | None = None

  def to_dict(self) -> dict[str, Any]:
    return {
        "root_model": self.root_model,
        "max_depth": self.max_depth,
        "max_iterations": self.max_iterations,
        "backend": self.backend,
        "backend_kwargs": {
            k: _serialize_value(v) for k, v in self.backend_kwargs.items()
        },
        "environment_type": self.environment_type,
        "environment_kwargs": {
            k: _serialize_value(v) for k, v in self.environment_kwargs.items()
        },
        "other_backends": self.other_backends,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "RLMMetadata":
    return cls(
        root_model=data.get("root_model", ""),
        max_depth=data.get("max_depth", 5),
        max_iterations=data.get("max_iterations", 30),
        backend=data.get("backend", ""),
        backend_kwargs=data.get("backend_kwargs", {}),
        environment_type=data.get("environment_type", ""),
        environment_kwargs=data.get("environment_kwargs", {}),
        other_backends=data.get("other_backends"),
    )


@dataclass
class QueryMetadata:
  """Metadata about the query context."""

  context_lengths: list[int]
  context_total_length: int
  context_type: str

  def __init__(
      self, prompt: str | list[str] | dict[Any, Any] | list[dict[Any, Any]]
  ):
    # Handle LazyFile and LazyFileCollection types
    # Import here to avoid circular imports
    try:
      from adk_rlm.files.lazy import LazyFile
      from adk_rlm.files.lazy import LazyFileCollection

      if isinstance(prompt, LazyFile):
        # Get file size without loading content if possible
        self.context_type = "lazy_file"
        try:
          self.context_lengths = [prompt.size_bytes or 0]
        except Exception:
          self.context_lengths = [0]
        self.context_total_length = sum(self.context_lengths)
        return
      elif isinstance(prompt, LazyFileCollection):
        self.context_type = "lazy_file_collection"
        self.context_lengths = []
        for f in prompt:
          try:
            self.context_lengths.append(f.size_bytes or 0)
          except Exception:
            self.context_lengths.append(0)
        self.context_total_length = sum(self.context_lengths)
        return
    except ImportError:
      pass

    if isinstance(prompt, str):
      self.context_lengths = [len(prompt)]
      self.context_type = "str"
    elif isinstance(prompt, dict):
      self.context_type = "dict"
      self.context_lengths = []
      for chunk in prompt.values():
        if isinstance(chunk, str):
          self.context_lengths.append(len(chunk))
        else:
          try:
            import json

            self.context_lengths.append(len(json.dumps(chunk, default=str)))
          except Exception:
            self.context_lengths.append(len(repr(chunk)))
    elif isinstance(prompt, list):
      self.context_type = "list"
      if len(prompt) == 0:
        self.context_lengths = [0]
      elif isinstance(prompt[0], dict):
        if "content" in prompt[0]:
          self.context_lengths = [
              len(str(chunk.get("content", ""))) for chunk in prompt
          ]
        else:
          self.context_lengths = []
          for chunk in prompt:
            try:
              import json

              self.context_lengths.append(len(json.dumps(chunk, default=str)))
            except Exception:
              self.context_lengths.append(len(repr(chunk)))
      else:
        self.context_lengths = [len(str(chunk)) for chunk in prompt]
    else:
      raise ValueError(f"Invalid prompt type: {type(prompt)}")

    self.context_total_length = sum(self.context_lengths)
