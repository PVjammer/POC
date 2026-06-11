# Query Execution System - Implementation Design

## Overview

This system supports **two use cases** with shared core components:

### 1. Evaluation Mode (Final Validation)
Measure whether documentation context helps agents answer queries successfully.
- Runs on holdout query set
- One-shot execution (no iteration)
- Compares multiple scenarios (baseline/reference/generated)
- Outputs: comparison reports, success rates, latency metrics

### 2. Testing Mode (Iteration Loop)
Discover documentation gaps by letting agents iterate to success.
- Runs on test query set during CM refinement loop
- Allows iteration until success (to discover what's missing)
- Captures structured reflection traces for attribution
- Outputs: FeedbackItems for DocGenerator to update docs

**Shared infrastructure:** Agent, outcome classification, tool call tracking, MCP integration.

---

## Key Design Principles

1. **Cold start per query**: Fresh agent instance with no conversation history.
2. **Production-identical agent**: No mention of "evaluation" or "testing" in prompts.
3. **Metrics collection is invisible**: Via LangChain callbacks, not agent logic.
4. **Mode determines behavior**: Same core code, different configurations.

---

## Use Case Comparison

| Aspect | Evaluation Mode | Testing Mode |
|--------|-----------------|--------------|
| **Purpose** | Measure & compare scenarios | Find gaps, generate feedback |
| **Iteration** | One-shot only | Iterate until success |
| **Output** | Reports, metrics | FeedbackItems, reflection traces |
| **Query set** | Holdout (unseen) | Test set (can repeat) |
| **When used** | End of CM pipeline | During CM iteration loop |
| **Success metric** | Success rate, latency | Gaps found, feedback quality |

---

## Core Components

### 1. Data Models

Simple, flat structures for clarity:

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class Outcome(str, Enum):
    """Did the query succeed in getting real data?"""
    SUCCESS = "success"  # Got actual data back
    ERROR = "error"      # MCP tool returned an error message
    EMPTY = "empty"      # MCP tool returned no results


@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    name: str
    arguments: dict[str, Any]
    result: Any
    duration_ms: int
    timestamp: datetime

    @property
    def is_context_read(self) -> bool:
        return self.name == "read_context"

    @property
    def is_mcp(self) -> bool:
        return not self.is_context_read


@dataclass
class QueryExecution:
    """Complete record of executing one query."""
    query_id: str
    query_text: str
    scenario: str

    # Timing
    total_latency_ms: int

    # What happened
    tool_calls: list[ToolCall]
    final_answer: str

    # Classification
    outcome: Outcome
    outcome_reason: str

    # Optional judge evaluation
    judge_score: float | None = None
    judge_reasoning: str | None = None

    @property
    def mcp_call_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_mcp)

    @property
    def context_read_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_context_read)


@dataclass
class ScenarioSummary:
    """Aggregate metrics for one scenario."""
    name: str
    total_queries: int

    # Outcomes
    success_count: int
    error_count: int
    empty_count: int

    # Latency (ms)
    avg_latency: float
    min_latency: int
    max_latency: int
    p50_latency: int
    p90_latency: int

    # Tool usage
    avg_mcp_calls: float
    avg_context_reads: float

    # Judge scores (if used)
    avg_judge_score: float | None

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_queries if self.total_queries > 0 else 0.0


# --- Testing Mode Data Models ---

@dataclass
class AttemptRecord:
    """Record of a single query attempt within an execution."""
    query_text: str  # The SQL/query that was attempted
    result: Any
    outcome: Outcome
    reasoning: str  # Agent's explanation of why it tried this


@dataclass
class TestExecution(QueryExecution):
    """
    Extended execution record for Testing Mode.

    Captures the full journey from initial attempt through
    exploration to final success, enabling delta analysis.
    """
    # Initial attempt (based only on provided context)
    initial_attempt: AttemptRecord | None = None

    # Exploration phase (what the agent had to discover)
    exploration_tool_calls: list[ToolCall] = field(default_factory=list)

    # Final successful attempt (if different from initial)
    final_attempt: AttemptRecord | None = None

    # Analysis
    required_exploration: bool = False  # Did agent need to explore beyond docs?
    context_gaps: list[str] = field(default_factory=list)  # What was missing

    @property
    def was_one_shot(self) -> bool:
        """Did agent succeed on first attempt without exploration?"""
        return not self.required_exploration and self.outcome == Outcome.SUCCESS


@dataclass
class FeedbackItem:
    """
    Structured recommendation for documentation update.

    Output of the Feedback Analyzer, input to DocGenerator.update().
    """
    action: Literal["ADD", "REMOVE", "MODIFY"]
    target: str  # e.g., "tables/users.md", "gotchas.md#id-format"
    reason: str  # Why this change is needed
    content: str | None  # Proposed content for ADD/MODIFY
    source_query_id: str  # Which test query revealed this gap
    source_trace: str  # Link to reflection trace for debugging
    confidence: float = 1.0  # How confident are we this is the right fix?


@dataclass
class TestRunResult:
    """
    Result of running test queries in Testing Mode.

    Used by the CM iteration loop to decide whether to continue
    and by the Feedback Analyzer to generate FeedbackItems.
    """
    executions: list[TestExecution]

    # Aggregate metrics
    total_queries: int
    one_shot_success_count: int
    required_exploration_count: int
    failure_count: int

    # Generated feedback (if analyze=True)
    feedback_items: list[FeedbackItem] = field(default_factory=list)

    @property
    def one_shot_success_rate(self) -> float:
        return self.one_shot_success_count / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def should_continue_iteration(self) -> bool:
        """Are there gaps worth fixing?"""
        return self.required_exploration_count > 0 or self.failure_count > 0
```

### 2. Production Agent

A standard agent for production incident response. No evaluation awareness.

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from pathlib import Path


class IncidentResponseAgent:
    """
    Production agent for incident response.

    This class has NO knowledge of evaluation - it's the same code
    that would run in production.
    """

    def __init__(
        self,
        llm: ChatAnthropic,
        mcp_tools: list,  # Tools from MCP server
        context_dir: Path | None = None,
        system_name: str = "the target system",
    ):
        self.llm = llm
        self.context_dir = context_dir
        self.tools = self._build_tools(mcp_tools)
        self.agent_executor = self._build_agent(system_name)

    def _build_tools(self, mcp_tools: list) -> list:
        """Combine MCP tools with context reading tool if applicable."""
        tools = list(mcp_tools)

        if self.context_dir:
            tools.append(self._make_context_tool())

        return tools

    def _make_context_tool(self):
        """Create tool for reading documentation files."""
        context_dir = self.context_dir  # Capture for closure

        @tool
        def read_context(filename: str) -> str:
            """Read a documentation file. Use this to understand the system before querying."""
            path = context_dir / filename
            if not path.exists():
                available = [f.name for f in context_dir.glob("**/*.md")]
                return f"File '{filename}' not found. Available: {available}"
            return path.read_text()

        return read_context

    def _build_agent(self, system_name: str) -> AgentExecutor:
        """Create the LangChain agent executor."""

        # Build system prompt - mention available docs if context_dir exists
        system_parts = [
            f"You are an incident response agent for {system_name}.",
            "",
            "Your goal: Answer queries by executing the correct query on the first attempt.",
        ]

        if self.context_dir:
            # List available documentation
            doc_files = list(self.context_dir.glob("**/*.md"))
            if doc_files:
                file_list = "\n".join(f"  - {f.relative_to(self.context_dir)}" for f in doc_files)
                system_parts.extend([
                    "",
                    "Available documentation (use read_context tool to access):",
                    file_list,
                    "",
                    "Read relevant documentation before querying to understand the schema.",
                ])

        system_prompt = "\n".join(system_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,  # Quiet for production
            max_iterations=10,  # Reasonable limit
            return_intermediate_steps=True,  # Need this for metrics
        )

    def run(self, query: str) -> tuple[str, list]:
        """
        Execute a query. Returns (answer, intermediate_steps).

        Each call is independent - no conversation history.
        """
        result = self.agent_executor.invoke({"input": query})
        return result["output"], result.get("intermediate_steps", [])
```

### 3. Outcome Classifier

Determines if a query execution succeeded, failed, or returned empty results.
The key insight: check what the MCP tool actually returned, not just whether it threw an exception.

```python
class OutcomeClassifier:
    """
    Classify query outcomes based on what MCP tools returned.

    An MCP tool can "succeed" (no exception) but still return:
    - An error message (possibly in another language)
    - Empty results
    - Garbage

    We need to detect these cases.
    """

    # Patterns that indicate an error message (not real data)
    ERROR_PATTERNS = [
        # English
        "error", "exception", "invalid", "failed", "failure",
        "syntax", "unexpected", "denied", "forbidden", "unauthorized",
        "not found", "does not exist", "unknown",
        # Chinese (common in some systems)
        "错误", "异常", "无效", "失败",
        # SQL-specific
        "sql", "query", "statement",
    ]

    # Patterns that indicate empty results
    EMPTY_PATTERNS = [
        "no results", "no rows", "0 rows", "zero rows",
        "empty", "not found", "none found",
        "[]", "{}", "null",
    ]

    def classify(
        self,
        tool_calls: list[ToolCall],
        final_answer: str,
    ) -> tuple[Outcome, str]:
        """
        Classify the outcome. Returns (outcome, reason).
        """
        # Get MCP calls only (not context reads)
        mcp_calls = [tc for tc in tool_calls if tc.is_mcp]

        # No MCP calls = agent never queried the system
        if not mcp_calls:
            return Outcome.ERROR, "Agent never called MCP tools"

        # Check the last MCP call - that's the one with the "answer"
        last_mcp = mcp_calls[-1]

        # Check if result looks like an error
        if self._looks_like_error(last_mcp.result):
            return Outcome.ERROR, f"MCP result appears to be error: {self._truncate(last_mcp.result)}"

        # Check if result is empty
        if self._is_empty(last_mcp.result):
            return Outcome.EMPTY, "MCP returned empty results"

        # Got something that looks like real data
        return Outcome.SUCCESS, "MCP returned data"

    def _looks_like_error(self, result: Any) -> bool:
        """Heuristic: does this look like an error message?"""
        if result is None:
            return True

        result_str = str(result).lower()

        # Very short results with error-like words are probably errors
        # Real data tends to be longer or structured
        if len(result_str) < 1000:
            error_pattern_count = sum(
                1 for pattern in self.ERROR_PATTERNS
                if pattern in result_str
            )
            # If multiple error patterns present, probably an error
            if error_pattern_count >= 2:
                return True
            # Single pattern + short length = likely error
            if error_pattern_count >= 1 and len(result_str) < 200:
                return True

        return False

    def _is_empty(self, result: Any) -> bool:
        """Heuristic: is this an empty result?"""
        if result is None:
            return True

        # Handle structured data
        if isinstance(result, (list, tuple)) and len(result) == 0:
            return True
        if isinstance(result, dict) and len(result) == 0:
            return True

        result_str = str(result).strip().lower()

        if not result_str:
            return True

        # Check for empty patterns
        for pattern in self.EMPTY_PATTERNS:
            if result_str == pattern or result_str.startswith(pattern + " "):
                return True

        return False

    def _truncate(self, value: Any, max_len: int = 100) -> str:
        """Truncate for display."""
        s = str(value)
        return s[:max_len] + "..." if len(s) > max_len else s
```

### 4. Query Runner

Executes a single query and collects metrics. Uses LangChain's `intermediate_steps`
to capture tool calls without wrapping/mutating tools.

```python
import time
from datetime import datetime


class QueryRunner:
    """
    Runs a single query against an agent and collects metrics.

    Uses LangChain's intermediate_steps for tool call tracking -
    no need to wrap or mutate tools.
    """

    def __init__(self, classifier: OutcomeClassifier | None = None):
        self.classifier = classifier or OutcomeClassifier()

    def run(
        self,
        agent: IncidentResponseAgent,
        query_id: str,
        query_text: str,
        scenario: str,
    ) -> QueryExecution:
        """Execute query and return complete execution record."""

        start_time = time.time()
        start_dt = datetime.now()

        try:
            answer, intermediate_steps = agent.run(query_text)
            tool_calls = self._extract_tool_calls(intermediate_steps, start_dt)
            outcome, outcome_reason = self.classifier.classify(tool_calls, answer)

        except Exception as e:
            answer = ""
            tool_calls = []
            outcome = Outcome.ERROR
            outcome_reason = f"Exception: {e}"

        total_latency_ms = int((time.time() - start_time) * 1000)

        return QueryExecution(
            query_id=query_id,
            query_text=query_text,
            scenario=scenario,
            total_latency_ms=total_latency_ms,
            tool_calls=tool_calls,
            final_answer=answer,
            outcome=outcome,
            outcome_reason=outcome_reason,
        )

    def _extract_tool_calls(
        self,
        intermediate_steps: list,
        start_dt: datetime,
    ) -> list[ToolCall]:
        """
        Extract ToolCall records from LangChain intermediate_steps.

        intermediate_steps is list of (AgentAction, result) tuples.
        """
        tool_calls = []

        for action, result in intermediate_steps:
            tool_calls.append(ToolCall(
                name=action.tool,
                arguments=action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input},
                result=result,
                duration_ms=0,  # LangChain doesn't track per-tool timing
                timestamp=start_dt,  # Approximate
            ))

        return tool_calls
```

---

## Testing Mode Components

The following components extend the core infrastructure to support the CM iteration loop.

### 4a. Reflective Agent (Testing Mode)

A variant of the production agent that explains its reasoning, enabling attribution.

```python
class ReflectiveAgent(IncidentResponseAgent):
    """
    Agent that explains its reasoning for testing/debugging.

    Used in Testing Mode to capture WHY the agent makes decisions,
    enabling the Feedback Analyzer to identify documentation gaps.
    """

    def _build_agent(self, system_name: str) -> AgentExecutor:
        """Create agent with reflection-enabled prompt."""

        system_parts = [
            f"You are an incident response agent for {system_name}.",
            "",
            "Your goal: Answer queries by executing the correct query on the first attempt.",
            "",
            "IMPORTANT: Think through your approach before each action:",
            "1. Explain what information you're looking for",
            "2. State what you found in the documentation (or that you couldn't find it)",
            "3. Explain your reasoning for the query you're about to try",
            "4. If a query fails, explain what you learned and what you'll try next",
        ]

        if self.context_dir:
            doc_files = list(self.context_dir.glob("**/*.md"))
            if doc_files:
                file_list = "\n".join(f"  - {f.relative_to(self.context_dir)}" for f in doc_files)
                system_parts.extend([
                    "",
                    "Available documentation (use read_context tool to access):",
                    file_list,
                    "",
                    "Read relevant documentation before querying to understand the schema.",
                ])

        system_prompt = "\n".join(system_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,  # Enable for reflection capture
            max_iterations=15,  # Allow more iterations for exploration
            return_intermediate_steps=True,
        )
```

### 4b. Test Runner (Testing Mode)

Runs queries with iteration allowed, capturing the journey from initial attempt to success.

```python
class TestRunner:
    """
    Runs queries in Testing Mode: allows iteration, captures reflection.

    Unlike QueryRunner (evaluation mode), this:
    - Allows the agent to iterate until success
    - Captures initial attempt vs final attempt
    - Identifies what exploration was required
    - Extracts context gaps for feedback generation
    """

    def __init__(self, classifier: OutcomeClassifier | None = None):
        self.classifier = classifier or OutcomeClassifier()

    def run(
        self,
        agent: ReflectiveAgent,
        query_id: str,
        query_text: str,
    ) -> TestExecution:
        """Run query with iteration, return detailed execution record."""

        start_time = time.time()
        start_dt = datetime.now()

        try:
            answer, intermediate_steps = agent.run(query_text)
            tool_calls = self._extract_tool_calls(intermediate_steps, start_dt)

            # Analyze the execution journey
            analysis = self._analyze_journey(tool_calls, answer)

            outcome, outcome_reason = self.classifier.classify(tool_calls, answer)

        except Exception as e:
            answer = ""
            tool_calls = []
            analysis = {"initial_attempt": None, "exploration": [], "final_attempt": None, "gaps": []}
            outcome = Outcome.ERROR
            outcome_reason = f"Exception: {e}"

        total_latency_ms = int((time.time() - start_time) * 1000)

        return TestExecution(
            query_id=query_id,
            query_text=query_text,
            scenario="testing",
            total_latency_ms=total_latency_ms,
            tool_calls=tool_calls,
            final_answer=answer,
            outcome=outcome,
            outcome_reason=outcome_reason,
            # Testing-specific fields
            initial_attempt=analysis["initial_attempt"],
            exploration_tool_calls=analysis["exploration"],
            final_attempt=analysis["final_attempt"],
            required_exploration=len(analysis["exploration"]) > 0,
            context_gaps=analysis["gaps"],
        )

    def _analyze_journey(
        self,
        tool_calls: list[ToolCall],
        final_answer: str,
    ) -> dict:
        """
        Analyze the execution to identify initial attempt, exploration, and gaps.

        Heuristic:
        - First MCP call = initial attempt
        - MCP calls to schema/describe tools = exploration
        - Last successful MCP call = final attempt (if different from initial)
        - Exploration tools used = context gaps
        """
        mcp_calls = [tc for tc in tool_calls if tc.is_mcp]
        context_calls = [tc for tc in tool_calls if tc.is_context_read]

        if not mcp_calls:
            return {"initial_attempt": None, "exploration": [], "final_attempt": None, "gaps": []}

        # Categorize MCP calls
        query_calls = []  # Actual data queries
        exploration_calls = []  # Schema exploration

        for tc in mcp_calls:
            # Heuristic: exploration tools have "describe", "schema", "list" in name
            if any(x in tc.name.lower() for x in ["describe", "schema", "list", "show"]):
                exploration_calls.append(tc)
            else:
                query_calls.append(tc)

        initial_attempt = None
        final_attempt = None

        if query_calls:
            first_query = query_calls[0]
            initial_attempt = AttemptRecord(
                query_text=str(first_query.arguments),
                result=first_query.result,
                outcome=self.classifier.classify([first_query], "")[0],
                reasoning="Initial attempt based on documentation",
            )

            if len(query_calls) > 1:
                last_query = query_calls[-1]
                final_attempt = AttemptRecord(
                    query_text=str(last_query.arguments),
                    result=last_query.result,
                    outcome=self.classifier.classify([last_query], "")[0],
                    reasoning="Final attempt after exploration",
                )

        # Extract gaps: what did the agent have to explore?
        gaps = []
        for ec in exploration_calls:
            gaps.append(f"Had to call {ec.name}({ec.arguments}) - info not in docs")

        return {
            "initial_attempt": initial_attempt,
            "exploration": exploration_calls,
            "final_attempt": final_attempt,
            "gaps": gaps,
        }

    def _extract_tool_calls(self, intermediate_steps: list, start_dt: datetime) -> list[ToolCall]:
        """Extract ToolCall records from LangChain intermediate_steps."""
        tool_calls = []
        for action, result in intermediate_steps:
            tool_calls.append(ToolCall(
                name=action.tool,
                arguments=action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input},
                result=result,
                duration_ms=0,
                timestamp=start_dt,
            ))
        return tool_calls
```

### 4c. Tester (CM Integration)

Orchestrates test runs and generates feedback for the CM iteration loop.

```python
class Tester:
    """
    Runs test queries and generates feedback for the CM iteration loop.

    This is the "Tester" component from DesignConsiderations.md.
    """

    def __init__(
        self,
        llm,
        mcp_tools: list,
        system_name: str,
    ):
        self.llm = llm
        self.mcp_tools = mcp_tools
        self.system_name = system_name
        self.runner = TestRunner()

    def run(
        self,
        context_dir: Path,
        queries: list[dict],
        analyze: bool = True,
    ) -> TestRunResult:
        """
        Run all test queries against current documentation.

        Args:
            context_dir: Path to current generated documentation
            queries: List of test queries [{"id": ..., "text": ...}]
            analyze: Whether to generate FeedbackItems

        Returns:
            TestRunResult with executions and optional feedback
        """
        executions = []

        for query in queries:
            print(f"  Testing [{query['id']}]: {query['text'][:50]}...")

            # Fresh reflective agent for each query
            agent = ReflectiveAgent(
                llm=self.llm,
                mcp_tools=self.mcp_tools,
                context_dir=context_dir,
                system_name=self.system_name,
            )

            execution = self.runner.run(
                agent=agent,
                query_id=query["id"],
                query_text=query["text"],
            )

            executions.append(execution)

            # Log result
            if execution.was_one_shot:
                print(f"       ✓ One-shot success")
            elif execution.outcome == Outcome.SUCCESS:
                print(f"       ~ Success after exploration ({len(execution.exploration_tool_calls)} exploration calls)")
            else:
                print(f"       ✗ {execution.outcome.value}: {execution.outcome_reason}")

        # Compute aggregates
        result = TestRunResult(
            executions=executions,
            total_queries=len(executions),
            one_shot_success_count=sum(1 for e in executions if e.was_one_shot),
            required_exploration_count=sum(1 for e in executions if e.required_exploration),
            failure_count=sum(1 for e in executions if e.outcome != Outcome.SUCCESS),
        )

        # Generate feedback if requested
        if analyze:
            result.feedback_items = self._generate_feedback(executions)

        return result

    def _generate_feedback(self, executions: list[TestExecution]) -> list[FeedbackItem]:
        """
        Analyze executions and generate FeedbackItems.

        This is a simple heuristic version. Could be enhanced with LLM analysis.
        """
        feedback = []

        for execution in executions:
            if not execution.required_exploration:
                continue  # No gaps to report

            for gap in execution.context_gaps:
                feedback.append(FeedbackItem(
                    action="ADD",
                    target="gaps_to_investigate",  # Placeholder - needs smarter targeting
                    reason=gap,
                    content=None,  # Feedback Analyzer will determine content
                    source_query_id=execution.query_id,
                    source_trace=self._format_trace(execution),
                ))

        return feedback

    def _format_trace(self, execution: TestExecution) -> str:
        """Format execution as reflection trace for debugging."""
        lines = [
            f"# Query: {execution.query_text}",
            "",
            "## Initial Attempt",
        ]

        if execution.initial_attempt:
            lines.extend([
                f"Query: {execution.initial_attempt.query_text}",
                f"Result: {execution.initial_attempt.outcome.value}",
                f"Reasoning: {execution.initial_attempt.reasoning}",
            ])
        else:
            lines.append("No initial attempt recorded")

        if execution.exploration_tool_calls:
            lines.extend(["", "## Exploration"])
            for tc in execution.exploration_tool_calls:
                lines.append(f"- {tc.name}({tc.arguments})")

        if execution.final_attempt and execution.final_attempt != execution.initial_attempt:
            lines.extend([
                "",
                "## Final Attempt",
                f"Query: {execution.final_attempt.query_text}",
                f"Result: {execution.final_attempt.outcome.value}",
            ])

        lines.extend([
            "",
            "## Context Gaps Identified",
        ])
        for gap in execution.context_gaps:
            lines.append(f"- {gap}")

        return "\n".join(lines)
```

---

### 5. LLM-as-Judge

Sanity check: does the answer look like it addresses the query?

**Important:** This is NOT evaluating correctness (we don't have ground truth).
It's checking whether the agent produced something reasonable vs. confused output.

```python
import json
from langchain_core.messages import HumanMessage


class Judge:
    """
    LLM-based sanity check for answers.

    NOT evaluating correctness - just whether the answer
    appears to address the query with actual data.
    """

    PROMPT_TEMPLATE = """You are evaluating an incident response agent's answer.

**User Query:** {query}

**Agent's Answer:** {answer}

**Outcome:** {outcome} ({outcome_reason})

Evaluate whether the answer appears to address the query:

- Score 1.0: Answer contains specific data/results that address the query
- Score 0.7: Answer addresses the query but data seems incomplete
- Score 0.4: Answer is vague or only loosely related to the query
- Score 0.0: Answer doesn't address the query, is confused, or admits failure

You are NOT judging correctness (you don't know the true answer).
You ARE judging whether this looks like a real, relevant response.

Respond ONLY with JSON:
{{"score": <float>, "reasoning": "<one sentence>"}}"""

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, execution: QueryExecution) -> tuple[float, str]:
        """
        Evaluate an execution. Returns (score, reasoning).

        Only call this for non-error outcomes - no point judging error messages.
        """
        if execution.outcome == Outcome.ERROR:
            return 0.0, "Skipped - execution resulted in error"

        prompt = self.PROMPT_TEMPLATE.format(
            query=execution.query_text,
            answer=execution.final_answer,
            outcome=execution.outcome.value,
            outcome_reason=execution.outcome_reason,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            return result["score"], result["reasoning"]
        except Exception as e:
            return 0.5, f"Judge failed to evaluate: {e}"
```

### 6. Evaluator

Orchestrates running queries across scenarios and collecting results.

```python
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
import json
import yaml


@dataclass
class ScenarioConfig:
    """Configuration for one evaluation scenario."""
    name: str
    description: str
    context_dir: Path | None  # None = no context (baseline)


@dataclass
class EvaluatorConfig:
    """Full evaluation configuration."""
    # Target system
    system_name: str
    mcp_tools: list  # Pre-loaded MCP tools

    # What to test
    scenarios: list[ScenarioConfig]
    queries: list[dict]  # [{"id": "q1", "text": "..."}, ...]

    # LLM settings
    llm: Any  # Pre-configured LLM instance
    judge_llm: Any | None = None  # None = skip judging

    # Output
    output_dir: Path = Path("eval_results")


class Evaluator:
    """
    Runs evaluation across multiple scenarios.

    For each (scenario, query) pair:
    1. Create fresh agent (cold start)
    2. Run query
    3. Classify outcome
    4. Optionally run judge
    5. Save results
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.runner = QueryRunner()
        self.judge = Judge(config.judge_llm) if config.judge_llm else None
        self.results: list[QueryExecution] = []

    def run(self) -> dict[str, ScenarioSummary]:
        """Run all scenarios and return summaries."""
        self._setup_output_dir()

        for scenario in self.config.scenarios:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario.name}")
            print(f"{'='*60}")

            for query in self.config.queries:
                self._run_single(scenario, query)

        summaries = self._compute_summaries()
        self._save_results(summaries)

        return summaries

    def _run_single(self, scenario: ScenarioConfig, query: dict):
        """Run one query in one scenario."""
        print(f"  [{query['id']}] {query['text'][:50]}...")

        # Fresh agent for each query (cold start)
        agent = IncidentResponseAgent(
            llm=self.config.llm,
            mcp_tools=self.config.mcp_tools,
            context_dir=scenario.context_dir,
            system_name=self.config.system_name,
        )

        # Execute
        execution = self.runner.run(
            agent=agent,
            query_id=query["id"],
            query_text=query["text"],
            scenario=scenario.name,
        )

        # Judge (if configured and not an error)
        if self.judge and execution.outcome != Outcome.ERROR:
            score, reasoning = self.judge.evaluate(execution)
            execution.judge_score = score
            execution.judge_reasoning = reasoning

        self.results.append(execution)

        # Log outcome
        status = "✓" if execution.outcome == Outcome.SUCCESS else "✗"
        print(f"       {status} {execution.outcome.value} | {execution.total_latency_ms}ms | "
              f"{execution.mcp_call_count} MCP calls")

    def _compute_summaries(self) -> dict[str, ScenarioSummary]:
        """Compute aggregate summaries per scenario."""
        summaries = {}

        for scenario in self.config.scenarios:
            executions = [e for e in self.results if e.scenario == scenario.name]
            summaries[scenario.name] = self._summarize(scenario.name, executions)

        return summaries

    def _summarize(self, name: str, executions: list[QueryExecution]) -> ScenarioSummary:
        """Compute summary statistics for a scenario."""
        if not executions:
            raise ValueError(f"No executions for scenario {name}")

        latencies = [e.total_latency_ms for e in executions]
        sorted_latencies = sorted(latencies)
        n = len(executions)

        judge_scores = [e.judge_score for e in executions if e.judge_score is not None]

        return ScenarioSummary(
            name=name,
            total_queries=n,
            success_count=sum(1 for e in executions if e.outcome == Outcome.SUCCESS),
            error_count=sum(1 for e in executions if e.outcome == Outcome.ERROR),
            empty_count=sum(1 for e in executions if e.outcome == Outcome.EMPTY),
            avg_latency=mean(latencies),
            min_latency=min(latencies),
            max_latency=max(latencies),
            p50_latency=sorted_latencies[n // 2],
            p90_latency=sorted_latencies[int(n * 0.9)],
            avg_mcp_calls=mean(e.mcp_call_count for e in executions),
            avg_context_reads=mean(e.context_read_count for e in executions),
            avg_judge_score=mean(judge_scores) if judge_scores else None,
        )

    def _setup_output_dir(self):
        """Create output directory structure."""
        self.run_dir = self.config.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        for scenario in self.config.scenarios:
            (self.run_dir / scenario.name / "traces").mkdir(parents=True, exist_ok=True)

    def _save_results(self, summaries: dict[str, ScenarioSummary]):
        """Save all results to disk."""
        # Save individual executions
        for execution in self.results:
            self._save_execution(execution)

        # Save scenario summaries
        for name, summary in summaries.items():
            summary_path = self.run_dir / name / "summary.json"
            summary_path.write_text(json.dumps(asdict(summary), indent=2))

        # Save overall comparison
        comparison = self._build_comparison(summaries)
        (self.run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
        (self.run_dir / "comparison.md").write_text(self._format_comparison_md(summaries))

        print(f"\nResults saved to: {self.run_dir}")

    def _save_execution(self, execution: QueryExecution):
        """Save one execution's results."""
        scenario_dir = self.run_dir / execution.scenario

        # Save metrics as JSON
        metrics_path = scenario_dir / f"{execution.query_id}_metrics.json"
        metrics_path.write_text(json.dumps(asdict(execution), indent=2, default=str))

        # Save trace as markdown (for human review)
        trace_path = scenario_dir / "traces" / f"{execution.query_id}_trace.md"
        trace_path.write_text(self._format_trace(execution))

    def _format_trace(self, execution: QueryExecution) -> str:
        """Format execution as human-readable trace."""
        lines = [
            f"# Query: {execution.query_text}",
            f"**Scenario:** {execution.scenario}",
            f"**Outcome:** {execution.outcome.value} - {execution.outcome_reason}",
            f"**Latency:** {execution.total_latency_ms}ms",
            "",
            "## Tool Calls",
        ]

        for i, tc in enumerate(execution.tool_calls, 1):
            tool_type = "context" if tc.is_context_read else "MCP"
            lines.append(f"### {i}. {tc.name} ({tool_type})")
            lines.append(f"**Arguments:** `{tc.arguments}`")
            lines.append(f"**Result preview:**")
            result_str = str(tc.result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            lines.append(f"```\n{result_str}\n```")
            lines.append("")

        lines.extend([
            "## Final Answer",
            execution.final_answer,
            "",
        ])

        if execution.judge_score is not None:
            lines.extend([
                "## Judge Evaluation",
                f"**Score:** {execution.judge_score}",
                f"**Reasoning:** {execution.judge_reasoning}",
            ])

        return "\n".join(lines)

    def _build_comparison(self, summaries: dict[str, ScenarioSummary]) -> dict:
        """Build comparison data structure."""
        return {
            "scenarios": {name: asdict(s) for name, s in summaries.items()},
            "best_latency": min(summaries.items(), key=lambda x: x[1].avg_latency)[0],
            "best_success_rate": max(summaries.items(), key=lambda x: x[1].success_rate)[0],
        }

    def _format_comparison_md(self, summaries: dict[str, ScenarioSummary]) -> str:
        """Format comparison as markdown report."""
        lines = [
            "# Evaluation Comparison Report",
            "",
            "## Summary",
            "",
            "| Scenario | Success Rate | Avg Latency | Avg MCP Calls | Judge Score |",
            "|----------|--------------|-------------|---------------|-------------|",
        ]

        for name, s in summaries.items():
            judge = f"{s.avg_judge_score:.2f}" if s.avg_judge_score else "N/A"
            lines.append(
                f"| {name} | {s.success_rate:.0%} ({s.success_count}/{s.total_queries}) | "
                f"{s.avg_latency:.0f}ms | {s.avg_mcp_calls:.1f} | {judge} |"
            )

        lines.extend([
            "",
            "## Latency Distribution",
            "",
            "| Scenario | Min | P50 | P90 | Max |",
            "|----------|-----|-----|-----|-----|",
        ])

        for name, s in summaries.items():
            lines.append(f"| {name} | {s.min_latency}ms | {s.p50_latency}ms | {s.p90_latency}ms | {s.max_latency}ms |")

        return "\n".join(lines)
```

### 7. Configuration

Simple YAML-based configuration loaded at startup.

```yaml
# eval_config.yaml

# Target system
system_name: "OmegaDB"
mcp_config: "config/omegadb_mcp.json"  # MCP server configuration

# Query set
queries: "queries/test_queries.json"

# LLM configuration
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  temperature: 0.0

# Judge (optional - set to null to skip)
judge_llm:
  provider: "anthropic"
  model: "claude-haiku-4-20250514"  # Cheaper model for judging
  temperature: 0.0

# Scenarios
scenarios:
  - name: "baseline"
    description: "No documentation context"
    context_dir: null

  - name: "reference"
    description: "Customer-provided documentation"
    context_dir: "context/reference"

  - name: "generated"
    description: "CM-generated documentation"
    context_dir: "context/generated"

# Output
output_dir: "eval_results"
```

```python
def load_config(path: str) -> EvaluatorConfig:
    """Load configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Load queries
    with open(raw["queries"]) as f:
        queries = json.load(f)["queries"]

    # Create LLM instances
    llm = create_llm(raw["llm"])
    judge_llm = create_llm(raw["judge_llm"]) if raw.get("judge_llm") else None

    # Load MCP tools (implementation depends on your MCP setup)
    mcp_tools = load_mcp_tools(raw["mcp_config"])

    # Build scenario configs
    scenarios = [
        ScenarioConfig(
            name=s["name"],
            description=s["description"],
            context_dir=Path(s["context_dir"]) if s.get("context_dir") else None,
        )
        for s in raw["scenarios"]
    ]

    return EvaluatorConfig(
        system_name=raw["system_name"],
        mcp_tools=mcp_tools,
        scenarios=scenarios,
        queries=queries,
        llm=llm,
        judge_llm=judge_llm,
        output_dir=Path(raw.get("output_dir", "eval_results")),
    )


def create_llm(config: dict):
    """Create LLM instance from config."""
    if config["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config["model"],
            temperature=config.get("temperature", 0.0),
        )
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")
```

### 8. Query Set Format

```json
{
  "name": "OmegaDB Test Queries",
  "queries": [
    {
      "id": "q001",
      "text": "Find all orders for the user with email test@example.com",
      "tags": ["join", "user-lookup"]
    },
    {
      "id": "q002",
      "text": "Get the total revenue for orders placed in the last 30 days",
      "tags": ["aggregation", "date-filter"]
    },
    {
      "id": "q003",
      "text": "List the top 10 customers by total order value",
      "tags": ["aggregation", "ranking"]
    }
  ]
}
```

Only `id` and `text` are required. `tags` are optional metadata for analysis.

---

## MCP Integration

**This is the part that needs to be adapted to your specific MCP setup.**

The evaluator expects MCP tools to be pre-loaded and passed in. How you load them depends on your MCP client:

```python
def load_mcp_tools(config_path: str) -> list:
    """
    Load tools from MCP server.

    This is a placeholder - implement based on your MCP setup.
    Options:
    1. Use mcp Python SDK to connect and list tools
    2. Manually define LangChain tools that call MCP
    3. Use langchain-mcp if available
    """
    # Option 1: Manual tool definition (simplest for now)
    from langchain_core.tools import tool

    @tool
    def execute_sql(query: str) -> str:
        """Execute a SQL query against the database."""
        # Call your MCP server here
        result = call_mcp_tool("execute_sql", {"query": query})
        return result

    @tool
    def describe_table(table_name: str) -> str:
        """Get schema information for a table."""
        result = call_mcp_tool("describe_table", {"table": table_name})
        return result

    return [execute_sql, describe_table]


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """
    Call an MCP tool.

    Implement this based on your MCP client setup.
    """
    # Your MCP client code here
    pass
```

The key point: MCP tools are just LangChain tools that happen to call an MCP server. The evaluator doesn't care about the transport.

---

## Output Structure

```
eval_results/
├── run_20250120_143000/
│   ├── comparison.json           # Machine-readable comparison
│   ├── comparison.md             # Human-readable report
│   │
│   ├── baseline/
│   │   ├── summary.json          # ScenarioSummary for this scenario
│   │   ├── q001_metrics.json     # QueryExecution for each query
│   │   ├── q002_metrics.json
│   │   └── traces/
│   │       ├── q001_trace.md     # Human-readable execution trace
│   │       └── q002_trace.md
│   │
│   ├── reference/
│   │   └── ...
│   │
│   └── generated/
│       └── ...
```

### Example: Query Execution (q001_metrics.json)

```json
{
  "query_id": "q001",
  "query_text": "Find all orders for the user with email test@example.com",
  "scenario": "generated",
  "total_latency_ms": 3421,
  "tool_calls": [
    {"name": "read_context", "arguments": {"filename": "tables/users.md"}, "result": "..."},
    {"name": "read_context", "arguments": {"filename": "tables/orders.md"}, "result": "..."},
    {"name": "execute_sql", "arguments": {"query": "SELECT o.* FROM orders o JOIN users u..."}, "result": "..."}
  ],
  "final_answer": "Found 5 orders for user test@example.com...",
  "outcome": "success",
  "outcome_reason": "MCP returned data",
  "judge_score": 1.0,
  "judge_reasoning": "Answer contains specific data that addresses the query"
}
```

### Example: Scenario Summary (summary.json)

```json
{
  "name": "generated",
  "total_queries": 20,
  "success_count": 18,
  "error_count": 1,
  "empty_count": 1,
  "avg_latency": 2847.0,
  "min_latency": 1200,
  "max_latency": 8500,
  "p50_latency": 2500,
  "p90_latency": 5000,
  "avg_mcp_calls": 1.15,
  "avg_context_reads": 1.8,
  "avg_judge_score": 0.92
}
```

### Example: Comparison Report (comparison.md)

```markdown
# Evaluation Comparison Report

## Summary

| Scenario | Success Rate | Avg Latency | Avg MCP Calls | Judge Score |
|----------|--------------|-------------|---------------|-------------|
| baseline | 45% (9/20) | 4521ms | 3.2 | 0.62 |
| reference | 90% (18/20) | 3104ms | 1.4 | 0.85 |
| generated | 90% (18/20) | 2847ms | 1.2 | 0.92 |

## Latency Distribution

| Scenario | Min | P50 | P90 | Max |
|----------|-----|-----|-----|-----|
| baseline | 2000ms | 4000ms | 7000ms | 12000ms |
| reference | 1500ms | 2800ms | 5000ms | 9000ms |
| generated | 1200ms | 2500ms | 5000ms | 8500ms |
```

---

## Integration with CM Pipeline

### How Components Map to DesignConsiderations.md

```
DesignConsiderations.md          This Design
─────────────────────────────    ─────────────────────────
Tester component            →    Tester class (Testing Mode)
  - Runs test queries            - TestRunner + ReflectiveAgent
  - Produces reflection traces   - TestExecution with gaps
  - Feeds Feedback Analyzer      - Outputs FeedbackItems

Evaluator component         →    Evaluator class (Evaluation Mode)
  - Runs holdout queries         - QueryRunner + IncidentResponseAgent
  - Measures success rate        - ScenarioSummary
  - Final validation             - Comparison reports
```

### CM Pipeline Integration

```python
# In the CM Pipeline (from DesignConsiderations.md)

class Pipeline:
    def __init__(self, ...):
        # Testing Mode - for iteration loop
        self.tester = Tester(
            llm=self.llm,
            mcp_tools=self.mcp_tools,
            system_name=config.system_name,
        )

        # Evaluation Mode - for final validation
        self.evaluator = Evaluator(config)

    def run(self, config: PipelineConfig) -> PipelineResult:
        # ... research, exploration, initial generation ...

        # Phase 4: Iteration Loop (uses Tester)
        for i in range(config.max_iterations):
            test_result = self.tester.run(
                context_dir=docs_path,
                queries=config.test_queries,
                analyze=True,  # Generate FeedbackItems
            )
            self.cache.write_test_run(test_result, run=i)

            if test_result.one_shot_success_rate >= config.success_threshold:
                print(f"Target reached: {test_result.one_shot_success_rate:.0%} one-shot success")
                break

            # Feed results to Feedback Analyzer (or use built-in feedback)
            feedback = test_result.feedback_items
            # Or: feedback = self.feedback_analyzer.analyze(test_result, docs)

            docs = self.doc_generator.update(docs, feedback)

        # Phase 5: Final Evaluation (uses Evaluator)
        eval_result = self.evaluator.run()  # Runs on holdout set

        return PipelineResult(final_docs=docs, eval_result=eval_result)
```

### Key Differences in Usage

| Use Case | Component | Queries | Output |
|----------|-----------|---------|--------|
| Iteration loop | `Tester` | test_queries.json | `TestRunResult` with `FeedbackItems` |
| Final validation | `Evaluator` | holdout_queries.json | `ScenarioSummary` + comparison.md |

---

## Implementation Checklist

### Phase 1: Core Components (Shared)

**Project Setup:**
- [ ] Create project structure (`src/cm/evaluation/`, `tests/`, `queries/`, `context/`)
- [ ] Set up dependencies (langchain, pyyaml, dataclasses)
- [ ] Configure environment (.env for API keys)

**Data Models:**
- [ ] `ToolCall` - single tool invocation
- [ ] `Outcome` enum - SUCCESS/ERROR/EMPTY
- [ ] `QueryExecution` - basic execution record
- [ ] `OutcomeClassifier` - classify MCP results

**Production Agent:**
- [ ] `IncidentResponseAgent` class
  - [ ] System prompt builder
  - [ ] `read_context` tool
  - [ ] LangChain agent executor
  - [ ] Cold start per query

**Query Runner:**
- [ ] `QueryRunner` class (evaluation mode)
  - [ ] Extract tool calls from intermediate_steps
  - [ ] Latency measurement
  - [ ] Outcome classification

### Phase 2: Testing Mode (CM Integration)

**Testing Data Models:**
- [ ] `AttemptRecord` - single query attempt
- [ ] `TestExecution` - extended with initial/final/gaps
- [ ] `FeedbackItem` - structured doc update recommendation
- [ ] `TestRunResult` - aggregated test results

**Testing Components:**
- [ ] `ReflectiveAgent` - agent that explains reasoning
- [ ] `TestRunner` - allows iteration, captures journey
- [ ] `Tester` - orchestrates test runs, generates feedback

**Integration:**
- [ ] Trace format matches DesignConsiderations.md
- [ ] FeedbackItem structure matches Feedback Analyzer input

### Phase 3: Evaluation Mode (Final Validation)

**Evaluation Data Models:**
- [ ] `ScenarioConfig` - scenario definition
- [ ] `ScenarioSummary` - aggregate metrics
- [ ] `EvaluatorConfig` - full config

**Evaluation Components:**
- [ ] `Evaluator` class
- [ ] `Judge` class (optional LLM sanity check)
- [ ] Comparison report generation

**Configuration:**
- [ ] YAML config parser
- [ ] Query set loader
- [ ] MCP tools loader

**CLI:**
- [ ] `python -m cm.evaluation run <config>` for standalone evaluation
- [ ] Integration with CM pipeline for iteration loop

### Phase 4: Polish

- [ ] End-to-end test with real queries
- [ ] Verify Tester → FeedbackItem → DocGenerator flow
- [ ] Verify Evaluator comparison reports

### Optional Enhancements

- [ ] LLM-based Feedback Analyzer (smarter than heuristics)
- [ ] Rich CLI output
- [ ] Parallel execution
- [ ] Cost tracking

---

## Technology Stack

**Language:** Python 3.10+

**Core Dependencies:**
- `langchain` - Agent framework and LLM abstraction
- `langchain-anthropic` - Anthropic LLM provider
- `pyyaml` - Config parsing
- `dataclasses` - Data models (stdlib)

**Optional:**
- `rich` - Pretty CLI output
- `pytest` - Testing
- `tenacity` - Retry logic for API calls

---

## Open Questions

1. **MCP tool loading**: How exactly to load tools from your MCP server?
   - Placeholder implementation provided - needs adaptation

2. **Exploration detection heuristics**: How to reliably identify "exploration" vs "query" MCP calls?
   - Current: Name-based heuristic (describe, schema, list, show)
   - May need tuning per target system

3. **Feedback quality**: Are heuristic-generated FeedbackItems good enough?
   - Start with heuristics, add LLM-based analysis if needed
   - Monitor whether DocGenerator can act on them

4. **Reflection capture**: Does the ReflectiveAgent prompt actually produce useful reasoning?
   - May need prompt tuning
   - Consider using extended thinking if available

5. **One-shot vs exploration success**: How to weight these in the iteration loop?
   - Currently: one_shot_success_rate is the target
   - Alternative: weight by how much exploration was needed

---

## Success Criteria

### For Testing Mode (CM Integration)

1. **Finds real gaps**: When docs are missing info, the Tester identifies it
2. **Actionable feedback**: FeedbackItems are specific enough for DocGenerator to act on
3. **Drives convergence**: Iteration loop improves one-shot success rate over time

### For Evaluation Mode (Final Validation)

1. **Measures the right thing**: Latency + outcome captures context quality
2. **Fair comparison**: Baseline/Reference/Generated compared consistently
3. **Actionable reports**: Clear which scenario wins and why

### For Both

1. **Shared infrastructure**: Same Agent, Classifier, ToolCall models
2. **Easy to run**: Single commands for testing or evaluation
3. **Debuggable**: Traces let you understand what went wrong

---

## CLI Usage

```bash
# --- Evaluation Mode (standalone comparison) ---
python -m cm.evaluation run eval_config.yaml
python -m cm.evaluation run eval_config.yaml --scenario generated  # Single scenario

# --- Testing Mode (part of CM pipeline) ---
python -m cm.testing run test_config.yaml  # Run test queries, output FeedbackItems
python -m cm.testing run test_config.yaml --query q001  # Single query debug

# --- Full CM Pipeline ---
python -m cm.pipeline run pipeline_config.yaml  # Research → Generate → Test → Iterate → Evaluate
```

---

## Alignment with DesignConsiderations.md

| DesignConsiderations Concept | Implementation |
|------------------------------|----------------|
| Cold start per query | ✓ Fresh agent instance each time |
| Reflection traces for attribution | ✓ `ReflectiveAgent` + `TestExecution.context_gaps` |
| FeedbackItem (ADD/REMOVE/MODIFY) | ✓ `FeedbackItem` dataclass |
| Tester in iteration loop | ✓ `Tester.run()` returns `TestRunResult` |
| Evaluator for holdout validation | ✓ `Evaluator.run()` returns `ScenarioSummary` |
| one_shot_success_rate metric | ✓ `TestRunResult.one_shot_success_rate` |
| success_threshold for convergence | ✓ `TestRunResult.should_continue_iteration` |
