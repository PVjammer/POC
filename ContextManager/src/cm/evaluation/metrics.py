"""
Metric collection for agent evaluation.

Extracts tool calls, latency, and outcome from conversation message histories.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json
import time


class Outcome(str, Enum):
    """Classification of query execution result."""
    SUCCESS = "success"  # Got real data back
    ERROR = "error"      # MCP returned an error message
    EMPTY = "empty"      # MCP returned no results


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""
    name: str
    arguments: dict[str, Any]
    result: Any = None

    @property
    def is_context_read(self) -> bool:
        """Is this a context/documentation read vs MCP call?"""
        # Customize this list based on your context-reading tool names
        context_tool_names = {"read_context", "read_file", "get_documentation"}
        return self.name in context_tool_names

    @property
    def is_mcp(self) -> bool:
        return not self.is_context_read


@dataclass
class ExecutionMetrics:
    """Metrics from a single query execution."""
    query: str
    latency_ms: int
    outcome: Outcome
    outcome_reason: str

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_answer: str | None = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    scenario: str | None = None  # e.g., "baseline", "with_docs"
    query_id: str | None = None

    @property
    def mcp_call_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_mcp)

    @property
    def context_read_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_context_read)

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "query_id": self.query_id,
            "scenario": self.scenario,
            "latency_ms": self.latency_ms,
            "outcome": self.outcome.value,
            "outcome_reason": self.outcome_reason,
            "mcp_call_count": self.mcp_call_count,
            "context_read_count": self.context_read_count,
            "total_tool_calls": self.total_tool_calls,
            "final_answer": self.final_answer,
            "timestamp": self.timestamp.isoformat(),
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "result_preview": str(tc.result)[:200]}
                for tc in self.tool_calls
            ],
        }


class OutcomeClassifier:
    """
    Classify whether an MCP result is success, error, or empty.

    Checks the actual content returned, not just whether an exception was raised.
    """

    # Patterns indicating an error (customize for your systems)
    ERROR_PATTERNS = [
        "error", "exception", "invalid", "failed", "failure",
        "syntax", "unexpected", "denied", "forbidden", "unauthorized",
        "not found", "does not exist", "unknown",
        # Chinese (common in some systems)
        "错误", "异常", "无效", "失败",
    ]

    # Patterns indicating empty results
    EMPTY_PATTERNS = [
        "no results", "no rows", "0 rows", "zero rows",
        "empty", "none found", "[]", "{}", "null",
    ]

    def classify(self, result: Any) -> tuple[Outcome, str]:
        """
        Classify a tool result. Returns (outcome, reason).
        """
        if result is None:
            return Outcome.EMPTY, "Result is None"

        result_str = str(result).lower().strip()

        if not result_str:
            return Outcome.EMPTY, "Result is empty string"

        # Check for error patterns
        if len(result_str) < 1000:  # Error messages tend to be short
            error_count = sum(1 for p in self.ERROR_PATTERNS if p in result_str)
            if error_count >= 2:
                return Outcome.ERROR, f"Multiple error patterns found in result"
            if error_count >= 1 and len(result_str) < 200:
                return Outcome.ERROR, f"Error pattern found in short result"

        # Check for empty patterns
        for pattern in self.EMPTY_PATTERNS:
            if result_str == pattern or result_str.startswith(pattern + " "):
                return Outcome.EMPTY, f"Result matches empty pattern: {pattern}"

        # Check structured empty
        if isinstance(result, (list, tuple)) and len(result) == 0:
            return Outcome.EMPTY, "Result is empty list"
        if isinstance(result, dict) and len(result) == 0:
            return Outcome.EMPTY, "Result is empty dict"

        return Outcome.SUCCESS, "Result contains data"


class MetricCollector:
    """
    Collect metrics from agent executions.

    Usage:
        collector = MetricCollector()

        # Option 1: Wrap a run
        with collector.measure("Find users by email") as ctx:
            messages = agent.run(query)
            ctx.set_messages(messages)

        # Option 2: Analyze existing messages
        metrics = collector.analyze(query, messages, latency_ms=1234)

        # Get results
        collector.summary()
        collector.to_json("results.json")
    """

    def __init__(self, classifier: OutcomeClassifier | None = None):
        self.classifier = classifier or OutcomeClassifier()
        self.executions: list[ExecutionMetrics] = []

    def measure(self, query: str, query_id: str | None = None, scenario: str | None = None):
        """
        Context manager for measuring a query execution.

        Usage:
            with collector.measure("Find users") as ctx:
                messages = agent.run(query)
                ctx.set_messages(messages)
        """
        return _MeasureContext(self, query, query_id, scenario)

    def analyze(
        self,
        query: str,
        messages: list[dict],
        latency_ms: int,
        query_id: str | None = None,
        scenario: str | None = None,
    ) -> ExecutionMetrics:
        """
        Analyze a message history and record metrics.

        Args:
            query: The original query text
            messages: List of message dicts from the conversation
            latency_ms: Total execution time
            query_id: Optional identifier for this query
            scenario: Optional scenario name (e.g., "baseline", "with_docs")

        Returns:
            ExecutionMetrics for this execution
        """
        tool_calls = self._extract_tool_calls(messages)
        final_answer = self._extract_final_answer(messages)
        outcome, outcome_reason = self._classify_execution(tool_calls)

        metrics = ExecutionMetrics(
            query=query,
            query_id=query_id,
            scenario=scenario,
            latency_ms=latency_ms,
            outcome=outcome,
            outcome_reason=outcome_reason,
            tool_calls=tool_calls,
            final_answer=final_answer,
        )

        self.executions.append(metrics)
        return metrics

    def _extract_tool_calls(self, messages: list[dict]) -> list[ToolCallRecord]:
        """Extract tool calls from message history."""
        tool_calls = []
        pending_calls = {}  # tool_call_id -> ToolCallRecord

        for msg in messages:
            # Handle tool calls from assistant messages
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    call_id = tc.get("id")
                    record = ToolCallRecord(
                        name=tc.get("function", {}).get("name", tc.get("name", "unknown")),
                        arguments=self._parse_arguments(tc.get("function", {}).get("arguments", tc.get("arguments", {}))),
                    )
                    tool_calls.append(record)
                    if call_id:
                        pending_calls[call_id] = record

            # Handle tool results
            if msg.get("role") == "tool":
                call_id = msg.get("tool_call_id")
                content = msg.get("content")

                if call_id and call_id in pending_calls:
                    pending_calls[call_id].result = content
                elif tool_calls:
                    # Fallback: assign to most recent call without result
                    for tc in reversed(tool_calls):
                        if tc.result is None:
                            tc.result = content
                            break

        return tool_calls

    def _parse_arguments(self, args: Any) -> dict:
        """Parse arguments, handling string JSON."""
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {"raw": args}
        return args if isinstance(args, dict) else {"raw": args}

    def _extract_final_answer(self, messages: list[dict]) -> str | None:
        """Extract the final assistant response (not a tool call)."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                return msg.get("content")
        return None

    def _classify_execution(self, tool_calls: list[ToolCallRecord]) -> tuple[Outcome, str]:
        """Classify overall execution outcome based on tool results."""
        mcp_calls = [tc for tc in tool_calls if tc.is_mcp]

        if not mcp_calls:
            return Outcome.ERROR, "No MCP tools were called"

        # Check the last MCP call's result
        last_mcp = mcp_calls[-1]
        return self.classifier.classify(last_mcp.result)

    def summary(self) -> dict:
        """Get aggregate summary of all executions."""
        if not self.executions:
            return {"total": 0}

        n = len(self.executions)
        latencies = [e.latency_ms for e in self.executions]

        return {
            "total": n,
            "success_count": sum(1 for e in self.executions if e.outcome == Outcome.SUCCESS),
            "error_count": sum(1 for e in self.executions if e.outcome == Outcome.ERROR),
            "empty_count": sum(1 for e in self.executions if e.outcome == Outcome.EMPTY),
            "success_rate": sum(1 for e in self.executions if e.outcome == Outcome.SUCCESS) / n,
            "avg_latency_ms": sum(latencies) / n,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_mcp_calls": sum(e.mcp_call_count for e in self.executions) / n,
            "avg_context_reads": sum(e.context_read_count for e in self.executions) / n,
        }

    def summary_by_scenario(self) -> dict[str, dict]:
        """Get summary grouped by scenario."""
        scenarios = {}
        for e in self.executions:
            scenario = e.scenario or "default"
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(e)

        result = {}
        for scenario, execs in scenarios.items():
            n = len(execs)
            latencies = [e.latency_ms for e in execs]
            result[scenario] = {
                "total": n,
                "success_rate": sum(1 for e in execs if e.outcome == Outcome.SUCCESS) / n,
                "avg_latency_ms": sum(latencies) / n,
                "avg_mcp_calls": sum(e.mcp_call_count for e in execs) / n,
            }

        return result

    def print_summary(self):
        """Print a formatted summary table."""
        by_scenario = self.summary_by_scenario()

        print(f"\n{'Scenario':<15} {'Success':<10} {'Latency':<12} {'MCP Calls':<10}")
        print("-" * 47)

        for scenario, stats in by_scenario.items():
            print(
                f"{scenario:<15} "
                f"{stats['success_rate']:.0%:<10} "
                f"{stats['avg_latency_ms']:.0f}ms{'':<6} "
                f"{stats['avg_mcp_calls']:.1f}"
            )

    def to_json(self, path: str):
        """Save all execution metrics to JSON file."""
        data = {
            "summary": self.summary(),
            "by_scenario": self.summary_by_scenario(),
            "executions": [e.to_dict() for e in self.executions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Clear all recorded executions."""
        self.executions = []


class _MeasureContext:
    """Context manager for measuring query execution."""

    def __init__(self, collector: MetricCollector, query: str, query_id: str | None, scenario: str | None):
        self.collector = collector
        self.query = query
        self.query_id = query_id
        self.scenario = scenario
        self.messages = None
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def set_messages(self, messages: list[dict]):
        """Set the message history from the agent run."""
        self.messages = messages

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = int((time.time() - self.start_time) * 1000)

        if self.messages is not None:
            self.collector.analyze(
                query=self.query,
                messages=self.messages,
                latency_ms=latency_ms,
                query_id=self.query_id,
                scenario=self.scenario,
            )

        return False  # Don't suppress exceptions
