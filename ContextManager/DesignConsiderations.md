# Context Manager (CM) - Design Considerations

This document captures the design discussion and decisions for the Context Manager agent system.

## Project Overview

**Goal:** Build a general-purpose "context generating" agent that bootstraps documentation for arbitrary systems, enabling downstream incident response agents to query those systems efficiently without extensive exploration.

**Success Metric:** A fresh agent, given only the generated context, can construct and execute correct queries on the first attempt (one-shot success) without needing to call schema-exploration tools.

**Key Constraint:** This must work for ANY system accessible via MCP (databases, logging systems, APIs, etc.) without system-specific code. Configuration and input—not code changes—enable customization.

---

## Problem Statement

### The Core Problem

During incident response, agents waste time discovering:
- What systems exist and how they're connected
- Database schemas and table purposes
- Log formats and locations
- API contracts and service relationships

Pre-computed context should eliminate this exploration overhead.

### Why Existing Approaches Fall Short

| Approach | Gap |
|----------|-----|
| Schema introspection only | Describes structure, not usage |
| Auto-generated documentation | Often over-engineers (adds harmful details) |
| CMDBs / Service catalogs | Stale, lacks query semantics |
| RAG over existing docs | Retrieval failures, unstructured source material |

### Observed Failure Mode

In initial prototyping with a SQL database ("OmegaDB"):
- A DeepAgent generated documentation that passed visual inspection
- But agents using that documentation still failed to query correctly
- Failure cause: **over-engineering**—documentation included details (e.g., "IDs stored in hex format") that led agents to generate incorrect queries (`SELECT HEX(id)...`)
- Meanwhile, human-prompted documentation (also LLM-generated, but via interactive Cursor chat) worked well

**Key Insight:** The problem isn't lack of information—it's inclusion of harmful information. Documentation should be minimal and usage-focused, not exhaustive.

---

## Design Principles

### 1. Start Minimal, Expand Reluctantly

Generate deliberately sparse documentation initially. Only add information when testing proves it's necessary. Every addition must be justified by a specific failure.

### 2. Test Functional Success, Not Visual Quality

Documentation quality is measured by whether agents can one-shot queries, not by whether it "looks complete" to a human reviewer.

### 3. Learn from Experts, Don't Reinvent

Use existing documentation, code, and query patterns as source material. The system distills expert knowledge rather than generating from scratch.

### 4. Example-Centric Over Description-Centric

```
BAD (description-centric):
"The users table contains user accounts. The id column stores
a unique identifier in hexadecimal format..."

GOOD (example-centric):
"Find user by email: SELECT * FROM users WHERE email = 'x'"
```

Show what works. Don't explain implementation details that might mislead.

### 5. Gotchas Are First-Class

Negative examples ("don't do X, do Y instead") are as valuable as positive guidance. A dedicated "gotchas" section prevents common mistakes.

---

## Architecture Overview

### High-Level Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Research   │────▶│  Explorer   │────▶│ DocGenerator│
│  (external  │     │  (validates │     │  (minimal   │
│   sources)  │     │   & grounds)│     │   docs)     │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Evaluator  │◀────│  Feedback   │◀────│   Tester    │
│  (holdout   │     │  Analyzer   │     │  (test set, │
│   set)      │     │             │     │   iterate)  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ DocGenerator│
                    │  (update)   │
                    └─────────────┘
                          │
                          ▼
                    [Iteration Loop]
```

### Components

#### 1. Researcher
- **Purpose:** Gather information from external sources (Confluence, GitLab, existing docs, code)
- **Input:** MCP connections to research sources
- **Output:** Unstructured findings about the target system
- **Does NOT:** Access the target system directly

#### 2. Explorer (needs better name—"Validator"? "Grounder"?)
- **Purpose:** Validate research against reality, extract schema, discover what research missed
- **Input:** Target system MCP, research findings
- **Output:**
  - Validated/corrected research claims
  - Actual schema information
  - Sample data patterns
  - Discoveries not in research
- **Key Behavior:** Grounds speculation in truth

#### 3. DocGenerator
- **Purpose:** Produce minimal documentation from research + exploration
- **Modes:**
  - `generate`: Initial creation from inputs
  - `update`: Apply feedback-driven modifications
- **Invariant:** Only component that writes documentation
- **Philosophy:** Minimal by default, add only what's proven necessary

#### 4. Tester
- **Purpose:** Validate documentation by having a fresh agent attempt queries
- **Input:** Documentation, test queries, target system MCP
- **Output:** Pass/fail results, reflection traces
- **Key Behavior:** Agent "thinks out loud" (reflection) to enable attribution of failures
- **Iterates:** Runs repeatedly during refinement loop

#### 5. Feedback Analyzer
- **Purpose:** Diagnose test failures and produce actionable edit recommendations
- **Input:** Test results, reflection traces, current documentation
- **Output:** Structured feedback items (ADD/REMOVE/MODIFY with specific targets)
- **Approach:** LLM-based analysis of reflection traces (rules may supplement)

#### 6. Evaluator
- **Purpose:** Final validation on holdout query set
- **Input:** Final documentation, holdout queries, target system MCP
- **Output:** Success metrics, failure analysis
- **Key Difference from Tester:** Uses different query set, runs once at end

#### 7. Query Generator (Optional)
- **Purpose:** Generate test queries when human-provided queries unavailable
- **Modes:**
  - Adversarial: Target gaps between discovered and documented entities
  - Synthetic: Generate representative queries from schema understanding
- **Input:** Cache (discovered entities, documented entities, schema)
- **Output:** Query set

---

## Data Flow and Storage

### Cache Structure (Proposed)

```
cache/
├── research/
│   ├── source_{name}.json        # Raw findings per source
│   └── merged.json               # Deduplicated findings
│
├── exploration/
│   ├── schema.json               # Tables, columns, types, relationships
│   ├── samples.json              # Sample values
│   ├── validated_claims.json     # Research claims + validation status
│   └── discoveries.json          # Found but not in research
│
├── testing/
│   └── run_{N}/
│       ├── config.json           # Query set, docs version
│       ├── results.json          # Pass/fail per query
│       └── traces/
│           └── query_{M}.md      # Reflection trace per query
│
├── feedback/
│   └── run_{N}_feedback.json     # Edit recommendations
│
└── metadata/
    ├── discovered_entities.json  # Union of all found entities
    └── documented_entities.json  # What's in current docs
```

### Feedback Item Structure

```python
@dataclass
class FeedbackItem:
    action: Literal["ADD", "REMOVE", "MODIFY"]
    target: str          # e.g., "users.email_address", "gotchas.id_encoding"
    reason: str          # Why this change is needed
    content: str | None  # Proposed content for ADD/MODIFY
    source_trace: str    # Link to reflection trace
```

### Reflection Trace Format

```markdown
# Query: Find all orders for user with email 'test@example.com'

## Initial Attempt
Based on context, I believe I need to query users and orders tables.
Context says users has 'email' column.

Attempting: SELECT o.* FROM orders o JOIN users u ON o.user_id = u.id
WHERE u.email = 'test@example.com'

Result: Error - column 'email' does not exist

## Exploration
[Tool call: describe_table('users')]
Actual columns: id, email_address, name, created_at

The column is 'email_address', not 'email'. Context was incorrect.

## Final Attempt
SELECT o.* FROM orders o JOIN users u ON o.user_id = u.id
WHERE u.email_address = 'test@example.com'

Result: Success

## Summary
- Context gap: users table column name wrong ('email' vs 'email_address')
- Required exploration: describe_table('users')
```

---

## Configuration Model

The system should be configurable, not coded, for different target systems.

### Required Configuration

```yaml
# Target system (what we're documenting)
target:
  mcp_server: "path/to/target-mcp-config"
  name: "OmegaDB"
  type: "sql_database"  # Informational, not behavioral

# Research sources (where we gather existing knowledge)
research_sources:
  - name: "confluence"
    mcp_server: "path/to/confluence-mcp-config"
  - name: "gitlab"
    mcp_server: "path/to/gitlab-mcp-config"

# Query sets
queries:
  test: "path/to/test_queries.json"      # Or: generator: "adversarial"
  holdout: "path/to/holdout_queries.json"

# Iteration control
iteration:
  max_rounds: 5
  success_threshold: 0.9  # Stop if 90% one-shot success

# Output
output:
  docs_path: "output/docs/"
  cache_path: "cache/"
```

### Query Set Format

```json
{
  "queries": [
    {
      "id": "q001",
      "natural_language": "Find all orders for the user with email test@example.com",
      "expected_tables": ["users", "orders"],  // Optional, for validation
      "tags": ["join", "user-lookup"]          // Optional, for analysis
    }
  ]
}
```

---

## Open Questions

### 1. Documentation Format

**Options considered:**
- Markdown: Human-readable, LLMs handle well, easy to edit
- JSON/YAML: Structured, easier programmatic updates, more compact
- Modular files: One file per entity vs. single file

**Current leaning:** Modular markdown files with an index. Optimizes for maintainability and debugging. Token savings from compact JSON are marginal (~20%).

**Proposed structure:**
```
docs/
├── _index.md           # Overview, entity list
├── tables/
│   ├── users.md
│   └── orders.md
├── relationships.md
├── patterns.md         # Common query patterns
└── gotchas.md
```

**Deferred decision:** May revisit based on what makes the update mechanism cleanest.

### 2. Test Query Source

**Priority order:**
1. Human-provided queries (most reliable)
2. Mined from existing docs/code (real usage patterns)
3. Adversarially generated (target discovered-but-undocumented entities)
4. Synthetically generated (from schema, least reliable)

**Open question:** How much can we rely on synthetic/adversarial generation? Need to test empirically.

### 3. Feedback Analyzer Implementation

**Options:**
- Pure LLM: Flexible, handles nuance, more expensive
- Pure rules: Fast, cheap, brittle
- Hybrid: Rules for common patterns, LLM for complex cases

**Current leaning:** Start with LLM-based, add rules for obvious patterns as optimization.

### 4. "Learning Session" Hybrid Approach

**Alternative approach discussed:** Instead of Research → Generate → Test, what if we:
1. Let an agent answer 50+ representative queries (building implicit knowledge)
2. Have an extraction agent analyze that session
3. Produce structured docs from successful patterns

**Status:** Worth considering as Phase 0 or alternative path. May produce more grounded initial docs.

### 5. Cross-System Context (V2)

**Goal:** Understand how multiple systems interact, how entities relate across systems.

**Challenges:**
- Entity resolution ("user_id" in DB = "userId" in logs?)
- Data flow mapping
- Service dependency graphs

**Status:** Deferred to V2. Current design should not preclude this.

---

## Abstractions Required

### LLM Client Interface

```python
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        system: str | None = None,
    ) -> LLMResponse: ...
```

Implementations for Anthropic, OpenAI, etc. Components never import SDK directly.

### MCP Client Interface

```python
class MCPClient(Protocol):
    def list_tools(self) -> list[ToolSpec]: ...
    def call_tool(self, name: str, arguments: dict) -> ToolResult: ...
```

Abstracts MCP transport details.

### Component Independence

Components should:
- Communicate only through defined data structures
- Read from cache, write to designated output locations
- Be testable in isolation with mocked inputs
- Not call each other directly (orchestrator manages flow)

---

## Pipeline Orchestration

```python
class Pipeline:
    def run(
        self,
        config: PipelineConfig,
    ) -> PipelineResult:
        # Phase 1: Research
        research = self.researcher.run(config.research_mcps)
        self.cache.write_research(research)

        # Phase 2: Exploration
        exploration = self.explorer.run(config.target_mcp, research)
        self.cache.write_exploration(exploration)

        # Phase 3: Initial Generation
        docs = self.doc_generator.generate(research, exploration)
        self.cache.write_docs(docs, version=0)

        # Phase 4: Iteration Loop
        for i in range(config.max_iterations):
            test_result = self.tester.run(config.target_mcp, docs, config.test_queries)
            self.cache.write_test_run(test_result, run=i)

            if test_result.success_rate >= config.success_threshold:
                break

            feedback = self.feedback_analyzer.analyze(test_result, docs)
            docs = self.doc_generator.update(docs, feedback)
            self.cache.write_docs(docs, version=i+1)

        # Phase 5: Final Evaluation
        eval_result = self.evaluator.run(config.target_mcp, docs, config.holdout_queries)

        return PipelineResult(final_docs=docs, eval_result=eval_result)
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Feedback loop doesn't converge | Max iteration limit, detect no-improvement cycles |
| Over-fitting to test queries | Holdout evaluation set |
| LLM costs during iteration | Start with small test sets, cache aggressively |
| Adversarial queries don't match real usage | Prioritize human-provided queries |
| Reflection traces too noisy for attribution | Structured trace format, explicit "summary" section |
| System changes invalidate cached data | Cache invalidation hooks, versioning |

---

## Next Steps

1. **Validate core loop:** Build minimal Researcher → Explorer → DocGenerator → Tester chain for OmegaDB
2. **Test feedback mechanism:** Can we reliably go from failed test → actionable edit?
3. **Measure baseline:** What's the one-shot success rate with current approach vs. no docs vs. human docs?
4. **Iterate on format:** Empirically determine what doc format produces best agent performance

---

## Appendix: Key Insights from Discussion

1. **Over-engineering is the enemy.** The failure mode is too much detail, not too little. Minimal docs + expansion beats comprehensive docs + pruning.

2. **Gotchas file is high-value.** Customer team specifically called this out. Negative examples prevent common mistakes.

3. **Reflection enables attribution.** Without "thinking out loud," we can't trace failures to specific documentation gaps.

4. **The delta matters.** Discovered-but-not-documented entities are candidates for adversarial testing.

5. **Schema alone isn't enough.** Raw schema describes structure, not usage. Effective docs show query patterns.

6. **Test against function, not form.** Visual inspection of docs is meaningless. Only one-shot query success matters.
