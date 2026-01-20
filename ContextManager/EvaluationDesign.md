# Evaluation System - Implementation Design

## Overview

**Goal:** Build a minimal evaluation system to measure agent query performance across different documentation contexts.

**Success Metrics (in priority order):**
1. **Minimal MCP calls**: Ideal = 1 call to execute correct query. More exploration calls = failure.
2. **Low latency**: Total time including context loading must be reasonable (target: <30s per query).
3. **Correctness**: Query returns actual results (not errors/empty), and answer makes sense (LLM-as-judge).

**Timeline:** 1-2 days for initial implementation.

---

## Key Design Principles

1. **Cold start per query**: Each query gets a fresh agent instance with no conversation history.
2. **Production-identical agent**: No mention of "evaluation" in prompts. Agent behaves exactly as it would in production.
3. **Evaluation is invisible**: Metrics collection happens in wrapper layers, not in agent logic.
4. **Scenario = script-level configuration**: Same agent code runs with different context availability.

---

## Evaluation Scenarios

Three configurations to compare (script-level, not agent-level):

1. **Baseline**: No context files available to agent
2. **Reference**: Customer-provided documentation files available
3. **Generated**: Our generated context files available

Each scenario uses:
- Same agent implementation (production code)
- Same query set
- Same target MCP server
- Same evaluation wrapper
- Different context directory paths

---

## Core Components

# 1. Production Agent (No Evaluation Logic)

A standard agent that will be used in production incident response:

**Capabilities:**
- Accepts natural language queries
- Has access to MCP server tools (target system)
- Has access to context reading tools (markdown files in designated directory)
- Executes queries and returns results

**System Prompt Structure:**
```
You are an incident response agent for [TARGET_SYSTEM].

You have access to:
1. Query execution tools for the target system
2. Documentation in the ./context directory

When answering queries:
1. Check relevant documentation first to understand schema/patterns
2. Construct and execute the correct query
3. Return results to the user

Be efficient - aim to execute the correct query on the first attempt.
```

Optionally, a small index file can be injected into the system prompt:
```
You are an incident response agent for [TARGET_SYSTEM].

# Available Documentation

[CONTENT OF _index.md]

Use the read_context tool to access specific documentation files as needed.
...
```

**Implementation** (pseudocode):
```python
class IncidentResponseAgent:
    """Production agent - no evaluation logic."""

    def __init__(
        self,
        llm: BaseChatModel,  # Langchain model
        mcp_tools: list[Tool],  # Tools from MCP server
        context_dir: str | None = None,  # Path to context docs
        index_content: str | None = None,  # Optional index to inject
    ):
        self.llm = llm
        self.tools = mcp_tools.copy()

        # Add context reading tool if context_dir provided
        if context_dir:
            self.tools.append(self._make_read_context_tool(context_dir))

        self.system_prompt = self._build_system_prompt(index_content)

    def _make_read_context_tool(self, context_dir: str) -> Tool:
        """Create a tool that reads markdown files from context_dir."""
        def read_context(filename: str) -> str:
            """Read a documentation file from the context directory."""
            path = Path(context_dir) / filename
            if not path.exists():
                return f"File {filename} not found"
            return path.read_text()

        return Tool(
            name="read_context",
            description="Read documentation files. Available files listed in system prompt.",
            func=read_context,
        )

    def query(self, user_query: str) -> str:
        """
        Execute a single query. This is a fresh conversation (no history).

        Returns the agent's final answer.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_query),
        ]

        # Use langchain agent executor
        response = self.agent_executor.invoke({"messages": messages})
        return response["output"]
```

**Key Design Points:**
- Agent has NO knowledge it's being evaluated
- Same code runs for all scenarios (baseline/reference/generated)
- Only difference between scenarios: which context_dir is provided
- Fresh conversation per query (no memory between queries)

# 2. Evaluation Wrapper (Metrics Collection)

Wraps the production agent to collect metrics invisibly:

```python
@dataclass
class ToolCall:
    name: str
    tool_type: Literal["mcp", "context"]  # Categorize tool calls
    arguments: dict
    result: Any
    timestamp: datetime
    duration_ms: int
    error: str | None = None

@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    query_id: str
    query_text: str

    # Performance metrics
    total_latency_ms: int
    mcp_tool_calls: int  # Number of calls to target system
    context_tool_calls: int  # Number of context reads
    total_tool_calls: int

    # Tool call details
    tool_calls: list[ToolCall]

    # Result quality
    result_type: Literal["success", "error", "empty"]  # Did we get data back?
    final_answer: str
    error_message: str | None

    # Full conversation trace
    conversation_trace: str  # For debugging

    # LLM-as-judge evaluation (populated later)
    judge_score: dict | None = None  # {answer_quality: 0-1, used_context: bool}

@dataclass
class EvaluationWrapper:
    """Wraps agent to collect metrics without changing behavior."""

    agent: IncidentResponseAgent

    def execute_query(self, query_id: str, query_text: str) -> QueryMetrics:
        """
        Execute query and collect metrics.

        Creates a fresh agent instance to ensure cold start.
        """
        start_time = time.time()

        # Intercept tool calls to track metrics
        tool_calls = []
        original_tools = self.agent.tools

        def wrap_tool(tool: Tool, tool_type: str) -> Tool:
            """Wrap a tool to track calls."""
            original_func = tool.func

            def tracked_func(*args, **kwargs):
                call_start = time.time()
                try:
                    result = original_func(*args, **kwargs)
                    error = None
                except Exception as e:
                    result = None
                    error = str(e)
                    raise
                finally:
                    tool_calls.append(ToolCall(
                        name=tool.name,
                        tool_type=tool_type,
                        arguments=kwargs,
                        result=result,
                        timestamp=datetime.now(),
                        duration_ms=int((time.time() - call_start) * 1000),
                        error=error,
                    ))
                return result

            tool.func = tracked_func
            return tool

        # Wrap tools with tracking
        for tool in self.agent.tools:
            tool_type = "context" if tool.name == "read_context" else "mcp"
            wrap_tool(tool, tool_type)

        # Execute query
        try:
            answer = self.agent.query(query_text)
            result_type = self._classify_result(answer, tool_calls)
            error_message = None
        except Exception as e:
            answer = ""
            result_type = "error"
            error_message = str(e)

        total_latency_ms = int((time.time() - start_time) * 1000)

        # Build conversation trace (for debugging/analysis)
        conversation_trace = self._build_trace(query_text, tool_calls, answer)

        return QueryMetrics(
            query_id=query_id,
            query_text=query_text,
            total_latency_ms=total_latency_ms,
            mcp_tool_calls=sum(1 for tc in tool_calls if tc.tool_type == "mcp"),
            context_tool_calls=sum(1 for tc in tool_calls if tc.tool_type == "context"),
            total_tool_calls=len(tool_calls),
            tool_calls=tool_calls,
            result_type=result_type,
            final_answer=answer,
            error_message=error_message,
            conversation_trace=conversation_trace,
        )

    def _classify_result(
        self,
        answer: str,
        tool_calls: list[ToolCall],
    ) -> Literal["success", "error", "empty"]:
        """Classify the result quality."""
        # Check if any MCP tool calls returned errors
        mcp_errors = [tc for tc in tool_calls if tc.tool_type == "mcp" and tc.error]
        if mcp_errors:
            return "error"

        # Check if result looks empty
        if not answer.strip() or "no results" in answer.lower():
            return "empty"

        return "success"

    def _build_trace(
        self,
        query: str,
        tool_calls: list[ToolCall],
        answer: str,
    ) -> str:
        """Build a human-readable trace for debugging."""
        lines = [f"# Query: {query}\n"]
        for i, tc in enumerate(tool_calls, 1):
            lines.append(f"## Tool Call {i}: {tc.name} ({tc.tool_type})")
            lines.append(f"Arguments: {tc.arguments}")
            lines.append(f"Duration: {tc.duration_ms}ms")
            if tc.error:
                lines.append(f"ERROR: {tc.error}")
            else:
                result_preview = str(tc.result)[:200]
                lines.append(f"Result: {result_preview}...")
            lines.append("")

        lines.append(f"## Final Answer\n{answer}")
        return "\n".join(lines)
```

# 3. LLM-as-Judge Evaluator

Separate component that evaluates query quality after execution:

```python
class LLMJudge:
    """Evaluates query results using an LLM."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def evaluate(
        self,
        query_text: str,
        answer: str,
        tool_calls: list[ToolCall],
    ) -> dict:
        """
        Evaluate a query result.

        Returns:
            {
                "answer_quality": 0.0-1.0,  # Does answer make sense?
                "answer_reasoning": str,
                "used_context": bool,  # Did agent read docs?
                "context_reasoning": str,
            }
        """
        context_reads = [tc for tc in tool_calls if tc.tool_type == "context"]

        prompt = f"""Evaluate this incident response interaction:

**User Query:** {query_text}

**Agent's Answer:** {answer}

**Tool Calls:** {len(tool_calls)} total ({len(context_reads)} context reads)

Please evaluate:

1. **Answer Quality** (0.0-1.0): Does the answer make sense given the query?
   - 1.0 = Directly answers the question with relevant data
   - 0.5 = Partial answer or unclear relevance
   - 0.0 = Nonsensical or completely wrong

2. **Context Usage**: Did the agent read documentation before querying?
   - Look for read_context tool calls in the trace

Respond in JSON format:
{{
    "answer_quality": <float>,
    "answer_reasoning": "<explanation>",
    "used_context": <bool>,
    "context_reasoning": "<explanation>"
}}
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)
```

---

# 4. Evaluation Runner

Orchestrates running a query set against different scenarios:

```python
@dataclass
class ScenarioConfig:
    """Configuration for one evaluation scenario."""
    name: str  # "baseline", "reference", "generated"
    description: str
    context_dir: str | None  # None for baseline
    index_file: str | None  # Optional index to inject into system prompt

@dataclass
class EvaluationConfig:
    """Full evaluation configuration."""
    target_mcp_config: str  # Path to MCP server config
    query_set_path: str
    scenarios: list[ScenarioConfig]
    llm_config: dict  # {provider: "anthropic", model: "...", temperature: 0}
    judge_llm_config: dict  # Config for LLM-as-judge
    output_dir: str

@dataclass
class ScenarioResult:
    """Results for one scenario."""
    scenario_name: str
    total_queries: int
    metrics: list[QueryMetrics]

    # Aggregate metrics
    avg_latency_ms: float
    avg_mcp_calls: float
    avg_context_calls: float
    avg_judge_score: float

    # Success categorization
    success_count: int  # result_type == "success"
    error_count: int
    empty_count: int

    # Efficiency metrics
    one_shot_success_count: int  # success with exactly 1 MCP call
    one_shot_success_rate: float

class EvaluationRunner:
    """Runs queries across multiple scenarios."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.query_set = self._load_query_set(config.query_set_path)
        self.llm = self._create_llm(config.llm_config)
        self.judge = LLMJudge(self._create_llm(config.judge_llm_config))

    def run_all_scenarios(self) -> dict[str, ScenarioResult]:
        """Run all scenarios and return results."""
        results = {}

        for scenario in self.config.scenarios:
            print(f"Running scenario: {scenario.name}")
            results[scenario.name] = self.run_scenario(scenario)

        return results

    def run_scenario(self, scenario: ScenarioConfig) -> ScenarioResult:
        """Run all queries for one scenario."""
        all_metrics = []

        for query in self.query_set.queries:
            print(f"  Query {query.id}: {query.text[:50]}...")

            # Create FRESH agent for this query (cold start)
            agent = self._create_agent(scenario)
            wrapper = EvaluationWrapper(agent)

            # Execute and collect metrics
            metrics = wrapper.execute_query(query.id, query.text)

            # Run LLM-as-judge evaluation
            if metrics.result_type != "error":
                judge_score = self.judge.evaluate(
                    query.text,
                    metrics.final_answer,
                    metrics.tool_calls,
                )
                metrics.judge_score = judge_score

            all_metrics.append(metrics)

            # Save individual result
            self._save_query_result(scenario.name, metrics)

        # Aggregate results
        return self._aggregate_results(scenario.name, all_metrics)

    def _create_agent(self, scenario: ScenarioConfig) -> IncidentResponseAgent:
        """Create a fresh agent instance for a query."""
        # Load MCP tools
        mcp_tools = self._load_mcp_tools(self.config.target_mcp_config)

        # Load index content if specified
        index_content = None
        if scenario.index_file:
            index_content = Path(scenario.index_file).read_text()

        return IncidentResponseAgent(
            llm=self.llm,
            mcp_tools=mcp_tools,
            context_dir=scenario.context_dir,
            index_content=index_content,
        )

    def _aggregate_results(
        self,
        scenario_name: str,
        metrics: list[QueryMetrics],
    ) -> ScenarioResult:
        """Aggregate metrics for a scenario."""
        return ScenarioResult(
            scenario_name=scenario_name,
            total_queries=len(metrics),
            metrics=metrics,
            avg_latency_ms=mean(m.total_latency_ms for m in metrics),
            avg_mcp_calls=mean(m.mcp_tool_calls for m in metrics),
            avg_context_calls=mean(m.context_tool_calls for m in metrics),
            avg_judge_score=mean(
                m.judge_score["answer_quality"]
                for m in metrics
                if m.judge_score
            ),
            success_count=sum(1 for m in metrics if m.result_type == "success"),
            error_count=sum(1 for m in metrics if m.result_type == "error"),
            empty_count=sum(1 for m in metrics if m.result_type == "empty"),
            one_shot_success_count=sum(
                1 for m in metrics
                if m.result_type == "success" and m.mcp_tool_calls == 1
            ),
            one_shot_success_rate=sum(
                1 for m in metrics
                if m.result_type == "success" and m.mcp_tool_calls == 1
            ) / len(metrics),
        )
```

# 5. Configuration System

Simple YAML-based configuration:

```yaml
# eval_config.yaml

# Target system
target:
  mcp_server_config: "path/to/omegadb-mcp-config.json"
  name: "OmegaDB"

# Query set
query_set: "queries/test_queries.json"

# LLM configuration
llm:
  provider: "anthropic"  # Used with langchain
  model: "claude-sonnet-4"
  temperature: 0.0  # Deterministic for evaluation
  api_key_env: "ANTHROPIC_API_KEY"

# LLM-as-judge configuration
judge_llm:
  provider: "anthropic"
  model: "claude-sonnet-4"  # Can use cheaper model if needed
  temperature: 0.0

# Scenarios to evaluate
scenarios:
  - name: "baseline"
    description: "No additional context"
    context_dir: null  # No context available
    index_file: null

  - name: "reference"
    description: "Customer-provided documentation"
    context_dir: "context/reference"  # Directory containing markdown files
    index_file: "context/reference/_index.md"  # Injected into system prompt

  - name: "generated"
    description: "Our generated context"
    context_dir: "context/generated"
    index_file: "context/generated/_index.md"

# Output
output:
  results_dir: "eval_results/"
  save_traces: true
```

**Notes:**
- `context_dir`: If provided, creates a `read_context(filename)` tool for the agent
- `index_file`: If provided, content is injected into system prompt (should be small!)
- Agent discovers what files exist via the index or trial-and-error

# 6. Query Set Format

Simple JSON format (provided externally):

```json
{
  "name": "OmegaDB Test Set",
  "version": "1.0",
  "queries": [
    {
      "id": "q001",
      "text": "Find all orders for the user with email test@example.com",
      "tags": ["join", "user-lookup"],
      "notes": "Tests cross-table joins and email lookup pattern"
    },
    {
      "id": "q002",
      "text": "Get the total revenue for orders placed in the last 30 days",
      "tags": ["aggregation", "date-filter"],
      "notes": "Tests temporal filtering and aggregation"
    }
  ]
}
```

**Fields:**
- `id`: Unique identifier for the query
- `text`: Natural language query (what agent receives)
- `tags`: Optional metadata for analysis
- `notes`: Optional human notes (not shown to agent)

---

## Implementation Details

### Context Access Pattern

The agent uses a **tool-based retrieval** approach:

1. **System prompt** contains:
   - Agent role and purpose
   - Optional index content (small! ~100-500 lines max)
   - Instruction to use `read_context` tool for documentation

2. **read_context tool**:
   - Takes filename as argument
   - Reads from designated context_dir
   - Returns file content or "not found"

3. **Agent decision-making**:
   - Agent decides which docs to read based on query
   - Can read multiple files if needed
   - Tracks as context_tool_calls in metrics

**Example System Prompt:**
```
You are an incident response agent for OmegaDB.

# Available Documentation

The following documentation is available via the read_context tool:
- tables/users.md
- tables/orders.md
- tables/products.md
- patterns.md (common query patterns)
- gotchas.md (important caveats)

Use read_context(filename) to access documentation before querying.

Your goal: Answer user queries efficiently by:
1. Reading relevant documentation
2. Executing the correct query on the first attempt
3. Returning results

Tools available:
- read_context(filename): Read documentation
- execute_sql(query): Execute SQL query on OmegaDB
```

### Tool Categorization

All tools are categorized for metrics:
- **context tools**: `read_context` (and any future doc-reading tools)
- **mcp tools**: Everything from the MCP server (execute_sql, etc.)

No filtering needed - we want to see if agents try to explore schema when docs are insufficient.

---

## Execution Flow

```
┌─────────────────────────────────────────────┐
│ 1. Load Configuration & Query Set          │
│    - Parse eval_config.yaml                 │
│    - Load queries from JSON                 │
│    - Initialize LLM clients                 │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 2. For Each Scenario                        │
│    (baseline → reference → generated)       │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 3. For Each Query in Set                    │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 4. Create Fresh Agent Instance              │
│    - Load MCP tools                         │
│    - Add read_context tool (if context_dir) │
│    - Build system prompt (+ index if any)   │
│    - NO conversation history                │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 5. Execute Query with Metrics Wrapper       │
│    - Start timer                            │
│    - Wrap tools to track calls              │
│    - Agent.query(user_query)                │
│    - Capture result                         │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 6. Collect Metrics                          │
│    - Total latency                          │
│    - MCP vs context tool call counts        │
│    - Result type (success/error/empty)      │
│    - Build conversation trace               │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 7. Run LLM-as-Judge (if not error)          │
│    - Evaluate answer quality                │
│    - Check if context was used              │
│    - Add judge_score to metrics             │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 8. Save Individual Result                   │
│    - Write metrics JSON                     │
│    - Write conversation trace MD            │
└─────────────────────────────────────────────┘
                    │
                    ▼
        [Repeat for all queries]
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 9. Aggregate Scenario Results               │
│    - Calculate avg metrics                  │
│    - Compute one-shot success rate          │
│    - Save scenario summary                  │
└─────────────────────────────────────────────┘
                    │
                    ▼
        [Repeat for all scenarios]
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 10. Generate Comparison Report              │
│     - Compare all scenarios                 │
│     - Identify performance differences      │
│     - Create visualizations/tables          │
└─────────────────────────────────────────────┘
```


---

## Output Structure

```
eval_results/
├── run_2025-01-20_14-30-00/
│   ├── config.yaml                    # Copy of eval config used
│   ├── summary.json                   # Aggregate results across scenarios
│   ├── comparison.md                  # Human-readable comparison report
│   │
│   ├── baseline/
│   │   ├── scenario_summary.json      # Aggregated metrics for scenario
│   │   ├── metrics/                   # Individual query metrics (JSON)
│   │   │   ├── q001_metrics.json
│   │   │   ├── q002_metrics.json
│   │   │   └── ...
│   │   └── traces/                    # Conversation traces (Markdown)
│   │       ├── q001_trace.md
│   │       ├── q002_trace.md
│   │       └── ...
│   │
│   ├── reference/
│   │   ├── scenario_summary.json
│   │   ├── metrics/
│   │   │   └── ...
│   │   └── traces/
│   │       └── ...
│   │
│   └── generated/
│       ├── scenario_summary.json
│       ├── metrics/
│       │   └── ...
│       └── traces/
│           └── ...
```

### Individual Query Metrics (metrics/q001_metrics.json)

```json
{
  "query_id": "q001",
  "query_text": "Find all orders for the user with email test@example.com",
  "total_latency_ms": 3421,
  "mcp_tool_calls": 1,
  "context_tool_calls": 2,
  "total_tool_calls": 3,
  "result_type": "success",
  "final_answer": "Found 5 orders for user test@example.com:\n- Order #1001...",
  "error_message": null,
  "tool_calls": [
    {
      "name": "read_context",
      "tool_type": "context",
      "arguments": {"filename": "tables/users.md"},
      "duration_ms": 145,
      "error": null
    },
    {
      "name": "read_context",
      "tool_type": "context",
      "arguments": {"filename": "tables/orders.md"},
      "duration_ms": 132,
      "error": null
    },
    {
      "name": "execute_sql",
      "tool_type": "mcp",
      "arguments": {
        "query": "SELECT o.* FROM orders o JOIN users u ON o.user_id = u.id WHERE u.email = 'test@example.com'"
      },
      "duration_ms": 3089,
      "error": null
    }
  ],
  "judge_score": {
    "answer_quality": 1.0,
    "answer_reasoning": "Query executed correctly and returned relevant results",
    "used_context": true,
    "context_reasoning": "Agent read both users and orders documentation before querying"
  }
}
```

### Scenario Summary (scenario_summary.json)

```json
{
  "scenario_name": "generated",
  "total_queries": 20,
  "avg_latency_ms": 2847,
  "avg_mcp_calls": 1.15,
  "avg_context_calls": 1.8,
  "avg_judge_score": 0.92,
  "success_count": 18,
  "error_count": 1,
  "empty_count": 1,
  "one_shot_success_count": 17,
  "one_shot_success_rate": 0.85
}
```

### Summary JSON (summary.json)

```json
{
  "run_id": "run_2025-01-20_14-30-00",
  "timestamp": "2025-01-20T14:30:00Z",
  "config_path": "eval_config.yaml",
  "query_set_path": "queries/test_queries.json",
  "total_queries": 20,
  "scenarios": {
    "baseline": {
      "avg_latency_ms": 4521,
      "avg_mcp_calls": 3.2,
      "avg_context_calls": 0.0,
      "avg_judge_score": 0.62,
      "one_shot_success_rate": 0.30
    },
    "reference": {
      "avg_latency_ms": 3104,
      "avg_mcp_calls": 1.4,
      "avg_context_calls": 2.1,
      "avg_judge_score": 0.85,
      "one_shot_success_rate": 0.70
    },
    "generated": {
      "avg_latency_ms": 2847,
      "avg_mcp_calls": 1.15,
      "avg_context_calls": 1.8,
      "avg_judge_score": 0.92,
      "one_shot_success_rate": 0.85
    }
  },
  "comparison": {
    "target_met": true,
    "generated_vs_reference": {
      "one_shot_success_delta": "+0.15",
      "latency_delta_ms": -257,
      "judge_score_delta": "+0.07"
    }
  }
}
```

### Comparison Report (comparison.md)

```markdown
# Evaluation Comparison Report

**Run ID:** run_2025-01-20_14-30-00
**Date:** 2025-01-20 14:30:00
**Query Set:** test_queries.json (20 queries)
**Target:** OmegaDB

---

## Executive Summary

| Scenario   | One-Shot Success | Avg Latency | Avg MCP Calls | Judge Score |
|------------|------------------|-------------|---------------|-------------|
| Baseline   | 30% (6/20)       | 4521ms      | 3.2           | 0.62        |
| Reference  | 70% (14/20)      | 3104ms      | 1.4           | 0.85        |
| Generated  | **85% (17/20)**  | **2847ms**  | **1.15**      | **0.92**    |

**Target Met:** ✓ (Generated exceeds Reference by +15pp on one-shot success)

**Key Findings:**
- Generated context reduces MCP calls by 64% vs baseline
- Generated context is 8% faster than reference docs
- Generated achieves one-shot success 85% of the time

---

## Detailed Metrics

### Efficiency (Lower is Better)

| Metric | Baseline | Reference | Generated | Generated vs Reference |
|--------|----------|-----------|-----------|------------------------|
| Avg MCP Calls | 3.2 | 1.4 | **1.15** | **-18%** |
| Avg Context Reads | 0.0 | 2.1 | **1.8** | **-14%** |
| Avg Total Latency | 4521ms | 3104ms | **2847ms** | **-8%** |

### Quality (Higher is Better)

| Metric | Baseline | Reference | Generated | Generated vs Reference |
|--------|----------|-----------|-----------|------------------------|
| One-Shot Success Rate | 30% | 70% | **85%** | **+15pp** |
| Overall Success Rate | 45% | 90% | **90%** | **+0pp** |
| Avg Judge Score | 0.62 | 0.85 | **0.92** | **+0.07** |

---

## Per-Query Breakdown

| ID | Query | Baseline MCP | Reference MCP | Generated MCP | Winner |
|----|-------|--------------|---------------|---------------|--------|
| q001 | Find orders for user email | 4 | 1 | **1** | TIE (R/G) |
| q002 | Total revenue last 30 days | 2 | 1 | **1** | TIE (R/G) |
| q003 | User signups by month | FAIL | 2 | **1** | **Generated** |
| q004 | Top 10 products | 1 | 1 | **1** | TIE (All) |
| ... | ... | ... | ... | ... | ... |

---

## Queries Where Generated Outperforms Reference

### q003: "Get user signup count by month for 2024"
- **Reference**: 2 MCP calls (explored schema first)
- **Generated**: 1 MCP call (correct query immediately)
- **Reason**: Generated docs include temporal aggregation pattern example

### q007: "Find users who placed orders in multiple countries"
- **Reference**: 3 MCP calls (trial and error on join conditions)
- **Generated**: 1 MCP call
- **Reason**: Generated gotchas.md explains multi-table join pattern

---

## Queries Where All Scenarios Struggled

### q015: "Find orders with shipping delay > 5 days"
- All scenarios: FAIL or 3+ MCP calls
- **Issue**: Requires computed field (shipped_date - ordered_date) not documented
- **Recommendation**: Add to gotchas or include calculated field examples

---

## Context Usage Analysis

| Scenario | Avg Context Reads | Most Read File | Read Frequency |
|----------|-------------------|----------------|----------------|
| Baseline | 0.0 | N/A | N/A |
| Reference | 2.1 | schema.md | 85% |
| Generated | 1.8 | gotchas.md | 90% |

**Insight:** Generated docs are more focused - agents find what they need faster.

---

## Recommendations

1. **Generated context is ready for production** - exceeds reference on all key metrics
2. **Investigate q015** - may need documentation on calculated fields
3. **gotchas.md is high-value** - 90% of queries referenced it, only 1.8 avg reads needed
4. **Reference docs could be improved** - longer and less focused than generated

---

## Next Steps

- Deploy generated context to production incident response agents
- Monitor real-world performance
- Iterate on q015 and similar edge cases
```

---

## Minimal Implementation Checklist

### Day 1: Core Agent & Metrics

**Project Setup:**
- [ ] Create project structure (`src/evaluation/`, `tests/`, `queries/`, `context/`)
- [ ] Set up dependencies (langchain, pyyaml, pydantic, anthropic)
- [ ] Configure environment (.env for API keys)

**Data Models:**
- [ ] Define `ToolCall` dataclass
- [ ] Define `QueryMetrics` dataclass
- [ ] Define `ScenarioResult` dataclass
- [ ] Define `ScenarioConfig` and `EvaluationConfig`

**Production Agent:**
- [ ] Implement `IncidentResponseAgent` class
  - [ ] System prompt builder
  - [ ] `read_context` tool implementation
  - [ ] Langchain agent executor setup
  - [ ] Fresh conversation per query

**Metrics Wrapper:**
- [ ] Implement `EvaluationWrapper` class
  - [ ] Tool call interception/tracking
  - [ ] Latency measurement
  - [ ] Tool categorization (MCP vs context)
  - [ ] Conversation trace builder
  - [ ] Result classification

**Testing:**
- [ ] Manual test with single query
- [ ] Verify cold start (no memory between queries)
- [ ] Verify tool call tracking works

### Day 2: Runner & Reporting

**Configuration:**
- [ ] YAML config parser
- [ ] JSON query set loader
- [ ] MCP client initialization (via langchain or SDK)

**LLM Judge:**
- [ ] Implement `LLMJudge` class
- [ ] Design evaluation prompt
- [ ] JSON response parsing

**Evaluation Runner:**
- [ ] Implement `EvaluationRunner` class
  - [ ] Scenario iteration
  - [ ] Query iteration with cold starts
  - [ ] Metrics collection
  - [ ] Individual result saving (JSON + MD)
  - [ ] Scenario aggregation

**Reporting:**
- [ ] Implement scenario summary generation
- [ ] Implement cross-scenario comparison
- [ ] Generate comparison.md report
- [ ] Save summary.json

**CLI:**
- [ ] Create CLI entry point (`python -m evaluation run <config>`)
- [ ] Add progress indicators (optional)
- [ ] Add error handling

**End-to-End Test:**
- [ ] Run full evaluation with test queries
- [ ] Verify all 3 scenarios execute
- [ ] Verify reports are generated correctly
- [ ] Manual review of results

### Optional Enhancements (if time):

- [ ] Rich CLI output with progress bars
- [ ] Retry logic for transient API failures
- [ ] Parallel query execution (with cold starts)
- [ ] Cost tracking (token usage per query)
- [ ] Interactive result viewer (web UI or TUI)

---

## Technology Stack

**Language:** Python 3.12+

**Core Dependencies:**
- `langchain` - Agent framework and LLM abstraction
- `langchain-anthropic` or `langchain-openai` - LLM provider
- `pyyaml` - Config parsing
- `pydantic` - Data validation and models

**Optional:**
- `rich` - Pretty CLI output with progress bars
- `pytest` - Testing framework
- `tenacity` - Retry logic for API calls

**MCP Integration:**
- Use langchain's tool abstraction to wrap MCP tools
- Can use `mcp` Python SDK if available, or langchain's `StructuredTool.from_function()`
- For V1: Manual tool wrapping is fine

---

## Open Questions

1. **Index file size limit**: How large can index files be before they hurt performance?
   - **Approach:** Start with 100-500 line limit, measure latency impact

2. **LLM-as-judge reliability**: Can we trust LLM evaluation for answer quality?
   - **Approach:** Run on subset, manually validate, adjust prompt if needed

3. **MCP tool wrapping**: Best way to convert MCP tools to langchain tools?
   - **Approach:** Start manual, automate if patterns emerge

4. **Parallel execution**: Run queries in parallel or sequential?
   - **Leaning:** Sequential for V1 (simpler, easier to debug), parallel later

5. **Non-determinism**: Should we run each query multiple times to measure variance?
   - **Leaning:** No for V1 (would 3x the cost), but use temperature=0

6. **Context directory structure**: Should agents be told about directory structure?
   - **Leaning:** Yes, include in index file (e.g., "tables/ contains table docs")

7. **Error recovery**: Should agents retry failed queries?
   - **Leaning:** No - we want to measure first-attempt success

---

## Success Criteria for This Evaluation System

The evaluation system itself is successful if:

1. **Reproducible**: Same query + scenario = same result
2. **Fast**: Can run 20 queries across 3 scenarios in <10 minutes
3. **Actionable**: Reports clearly show where generated context wins/loses
4. **Extensible**: Easy to add new scenarios or metrics later

---

## Next Steps

1. ✅ Review design document
2. Set up project structure
3. Implement Day 1 checklist
4. Implement Day 2 checklist
5. Run first full evaluation
6. Review results and iterate on context generation
