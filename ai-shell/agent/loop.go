// Package agent implements a clean multi-turn agent loop using native tool calling.
//
// Design:
//   - Uses ChatWithTools (non-streaming) for all rounds; tool call args must be complete
//   - Proper message sequence: system + history, no single-message packing
//   - Spinner during LLM wait, tool call display during execution
//   - Shell context (cwd, last cmd, exit code) injected into system prompt each turn
//   - Active context slots (from /ctx) injected into system prompt
//   - Loop-breaker: max rounds + repeated-identical-call detection
package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// ErrMaxRounds is returned when the agent exhausts its round budget. The loop
// delivers a forced partial response before returning this error so the caller
// always receives some output. The caller may re-invoke Run to continue.
var ErrMaxRounds = errors.New("reached max rounds")

const maxRounds = 25

// BaseSystemPrompt is the default system prompt for the agentic (!) mode.
// Exported so shell packages can use it when composing additional_instructions.
const BaseSystemPrompt = `You are an AI shell assistant running inside a Unix terminal.
Help the user accomplish tasks using shell commands and your own knowledge.
Be concise. Show relevant output. Prefer doing over explaining.
Use tools when needed; answer directly when you can.

Before your first tool call, state in one sentence what you are looking for and which
tool reaches it most directly. Once you have relevant results, synthesize your answer —
don't keep exploring beyond what is needed to answer the question.

For complex multi-step tasks, use task_list to track sub-goals: add tasks at the start,
mark them done as you complete each one. Skip it for simple lookups.

When context slots are shown as stubs in the "Active context" section, call
read_context() to retrieve their full content BEFORE exploring the filesystem.
After gathering information with tools, always provide a comprehensive text response.`

// AdvisorySystemPrompt is used for ? (read-only) queries.
// Advisory mode has access to read_context and describe_tool but not bash.
const AdvisorySystemPrompt = `You are an AI assistant embedded in a Unix terminal.
Answer questions, explain concepts, and guide the user on how to accomplish tasks.
Provide clear, concise explanations and concrete command examples in code blocks.
Do not attempt to execute shell commands — describe what to do instead.

When context slots are shown as stubs in the "Active context" section, call
read_context() to retrieve their full content before answering.`

// CtxSlot is a named context entry passed to the agent.
// Small slots (Description == "") are injected verbatim into the system prompt.
// Large slots carry a one-line Description stub; the agent uses read_context() to fetch content.
type CtxSlot struct {
	Content     string
	Description string // non-empty = stub mode
}

func (s CtxSlot) isStub() bool { return s.Description != "" }

// LoopConfig holds tuneable parameters for the agent loop.
type LoopConfig struct {
	MaxHistoryMessages int    // number of messages to keep in context (default 20)
	ToolOutputMaxChars int    // truncate/summarize tool results above this (default 4000)
	ToolOverflow       string // "truncate" or "summarize"

	// Phase 1 — tool output stripping.
	ToolOutputKeepRounds int // rounds of tool outputs to keep verbatim; older ones are stripped (default 3)

	// Phase 2 — token budget.
	MaxContextTokens int // model context ceiling in tokens; history trimmed to stay under 75% (default 8192)

	// Phase 3 — ctx slot auto-stub.
	CtxInlineThreshold int // bytes; slots larger than this are shown as stubs (default 4096)

	// Phase 4 — LLM compaction.
	CompactionThreshold    float64 // fire compaction at this fraction of MaxContextTokens (default 0.75)
	CompactionTailMessages int     // messages always kept verbatim in the protected tail (default 20)

	MaxResponseTokens int // max tokens the model may generate per turn (default 16384)
	MaxRounds         int // cap on tool-call rounds per Run() call (default 25)
}

func defaultLoopConfig() LoopConfig {
	return LoopConfig{
		MaxHistoryMessages:     20,
		ToolOutputMaxChars:     4000,
		ToolOverflow:           "truncate",
		ToolOutputKeepRounds:   3,
		MaxContextTokens:       8192,
		CtxInlineThreshold:     4096,
		CompactionThreshold:    0.75,
		CompactionTailMessages: 20,
		MaxResponseTokens:      16384,
	}
}

// ShellContext holds the current state of the shell, injected into the agent
// system prompt so the AI has accurate context for each turn.
type ShellContext struct {
	CWD          string
	LastCommand  string
	LastExitCode int
	LastStderr   string
}

// Loop is a stateful, multi-turn agent. One instance persists across user turns
// so conversation history is maintained for the session.
type Loop struct {
	provider       llm.ToolCallingProvider
	tools          []llm.ToolDef
	handlers       map[string]func(map[string]interface{}) (string, error)
	history        []llm.ChatMessage
	shellCtx       ShellContext
	contextSlots   map[string]CtxSlot
	cfg            LoopConfig
	spinnerEnabled bool
	systemPrompt   string // empty = use BaseSystemPrompt

	// Compaction state (Phase 4).
	compactionSummary string // current structured summary; empty = never compacted
	compactionDepth   int    // number of times compaction has run this session
	turnStart         int    // index in l.history of the current turn's user message; set by Run()

	// Per-turn task list. Lives outside history so it is never compacted.
	// Cleared at the start of each Run() call.
	taskList   []taskEntry
	nextTaskID int

	// Skill catalog and loader — set by the shell via SetSkillCatalog.
	// skillCatalog is pre-filtered for this agent's skill allowlist.
	// skillLoader satisfies skills.SkillLoader without importing the skills package.
	skillCatalog []SkillCatalogEntry
	skillLoader  interface {
		Body(name string) (string, error)
	}

	// Session querier — set by the shell when sessions are merged via /session merge.
	// Enables the query_session tool; nil = tool not available.
	sessionQuerier interface {
		QuerySession(ctx context.Context, name, question string) (string, error)
	}
	queryableSessions []string // session names available via query_session

	// Callbacks for the shell to display tool activity.
	OnToolCall   func(name string, args map[string]interface{})
	OnToolResult func(name string, result string)

	// debugLog receives a human-readable trace of every LLM call when non-nil.
	debugLog io.Writer
}

// New creates a new agent loop with default configuration.
func New(
	provider llm.ToolCallingProvider,
	tools []llm.ToolDef,
	handlers map[string]func(map[string]interface{}) (string, error),
) *Loop {
	return &Loop{
		provider:       provider,
		tools:          tools,
		handlers:       handlers,
		history:        make([]llm.ChatMessage, 0, 32),
		cfg:            defaultLoopConfig(),
		spinnerEnabled: true,
	}
}

// SetShellContext updates the shell state injected into every LLM call.
func (l *Loop) SetShellContext(ctx ShellContext) { l.shellCtx = ctx }

// SetContextSlots updates named context content injected into the system prompt.
func (l *Loop) SetContextSlots(slots map[string]CtxSlot) { l.contextSlots = slots }

// ToolDefs returns the current tool definitions (used by the describe_tool handler).
func (l *Loop) ToolDefs() []llm.ToolDef { return l.tools }

// GetConfig returns the current loop configuration.
func (l *Loop) GetConfig() LoopConfig { return l.cfg }

// SetConfig updates tuneable loop parameters.
func (l *Loop) SetConfig(cfg LoopConfig) {
	if cfg.MaxHistoryMessages > 0 {
		l.cfg.MaxHistoryMessages = cfg.MaxHistoryMessages
	}
	if cfg.ToolOutputMaxChars > 0 {
		l.cfg.ToolOutputMaxChars = cfg.ToolOutputMaxChars
	}
	if cfg.ToolOverflow != "" {
		l.cfg.ToolOverflow = cfg.ToolOverflow
	}
	if cfg.CtxInlineThreshold > 0 {
		l.cfg.CtxInlineThreshold = cfg.CtxInlineThreshold
	}
	if cfg.ToolOutputKeepRounds > 0 {
		l.cfg.ToolOutputKeepRounds = cfg.ToolOutputKeepRounds
	}
	if cfg.MaxContextTokens > 0 {
		l.cfg.MaxContextTokens = cfg.MaxContextTokens
	}
	if cfg.CompactionThreshold > 0 {
		l.cfg.CompactionThreshold = cfg.CompactionThreshold
	}
	if cfg.CompactionTailMessages > 0 {
		l.cfg.CompactionTailMessages = cfg.CompactionTailMessages
	}
	if cfg.MaxResponseTokens > 0 {
		l.cfg.MaxResponseTokens = cfg.MaxResponseTokens
	}
	if cfg.MaxRounds > 0 {
		l.cfg.MaxRounds = cfg.MaxRounds
	}
}

// SetTools replaces the tool list (called when functions are reloaded).
func (l *Loop) SetTools(tools []llm.ToolDef, handlers map[string]func(map[string]interface{}) (string, error)) {
	l.tools = tools
	l.handlers = handlers
}

// SetProvider replaces the LLM provider (called when the model is switched mid-session).
func (l *Loop) SetProvider(p llm.ToolCallingProvider) { l.provider = p }

// CopyHistory returns a deep copy of the conversation history.
// Used by session forking to snapshot history at the fork point.
func (l *Loop) CopyHistory() []llm.ChatMessage {
	cp := make([]llm.ChatMessage, len(l.history))
	copy(cp, l.history)
	return cp
}

// SetHistory replaces the conversation history. Used when initialising a
// forked session with the parent's history snapshot.
func (l *Loop) SetHistory(h []llm.ChatMessage) { l.history = h }

// SetDebugLog directs a human-readable trace of every LLM call to w.
// Pass nil to disable. The caller owns the writer's lifecycle.
func (l *Loop) SetDebugLog(w io.Writer) { l.debugLog = w }

// RunOneShot makes a single streaming LLM call with the given system and user
// messages. It does NOT read or write conversation history, does NOT trigger
// compaction, and does NOT use any tools. Designed for focused one-shot tasks
// (commit messages, summaries, etc.) that should not pollute the session.
func (l *Loop) RunOneShot(ctx context.Context, systemPrompt, userMsg string, onToken func(string)) error {
	msgs := []llm.ChatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userMsg},
	}
	sp := l.newSpinnerOrNop()
	ch := l.provider.ChatMessages(ctx, msgs, llm.DefaultOptions())
	sp.stop()
	for chunk := range ch {
		if chunk.Error != nil {
			return fmt.Errorf("llm: %w", chunk.Error)
		}
		if chunk.Text != "" && onToken != nil {
			onToken(chunk.Text)
		}
	}
	return nil
}

// SkillCatalogEntry is a minimal view of a skill for system prompt injection.
type SkillCatalogEntry struct {
	Name                   string
	Description            string
	DisableModelInvocation bool
}

// SetSkillCatalog wires the skill catalog and loader into the loop.
// catalog should already be filtered for this agent's skill allowlist.
// Pass nil loader to disable skill support.
func (l *Loop) SetSkillCatalog(catalog []SkillCatalogEntry, loader interface{ Body(string) (string, error) }) {
	l.skillCatalog = catalog
	l.skillLoader = loader
}

// SetSessionQuerier wires a session querier into the loop so the agent can call
// query_session() on sessions merged via /session merge. Pass nil to disable.
func (l *Loop) SetSessionQuerier(q interface {
	QuerySession(ctx context.Context, name, question string) (string, error)
}, sessions []string) {
	l.sessionQuerier = q
	l.queryableSessions = sessions
}

// SetSpinnerEnabled controls whether a spinner is shown during LLM waits.
// Disable for background execution to avoid corrupting the foreground terminal.
func (l *Loop) SetSpinnerEnabled(v bool) { l.spinnerEnabled = v }

// SetSystemPrompt overrides the default system prompt for the next Run call.
// Pass "" to restore the default agentic prompt.
func (l *Loop) SetSystemPrompt(p string) { l.systemPrompt = p }

func (l *Loop) newSpinnerOrNop() spinnerIface {
	if l.spinnerEnabled {
		return newSpinner()
	}
	return &noopSpinner{}
}

// Run executes one user turn — potentially many agent rounds.
// onToken is called with each piece of the final text response.
func (l *Loop) Run(ctx context.Context, userMsg string, onToken func(string)) error {
	l.turnStart = len(l.history) // record where this turn's user message will land
	defer func() { l.turnStart = 0 }()

	// Reset the task list for this invocation — it is ephemeral and per-query.
	l.taskList = nil
	l.nextTaskID = 1

	l.history = append(l.history, llm.ChatMessage{
		Role:    "user",
		Content: userMsg,
	})

	opts := &llm.CompletionOptions{
		MaxTokens:   l.cfg.MaxResponseTokens,
		Temperature: 0.3,
	}

	type callSig struct{ name, args string }
	recentCalls := make(map[callSig]int)
	nudges := 0

	sp := l.newSpinnerOrNop()

	rounds := maxRounds
	if l.cfg.MaxRounds > 0 {
		rounds = l.cfg.MaxRounds
	}
	for round := 0; round < rounds; round++ {
		if ctx.Err() != nil {
			sp.stop()
			return ctx.Err()
		}

		if l.shouldCompact() {
			sp.stop()
			fmt.Fprintf(os.Stderr, "\033[2m[compacting context…]\033[0m\n")
			if err := l.runCompaction(ctx); err != nil {
				fmt.Fprintf(os.Stderr, "\033[2m[compaction failed: %v]\033[0m\n", err)
			}
			sp = l.newSpinnerOrNop()
		}

		msgs := l.buildMessages()
		roundTools := append(l.tools, taskListToolDef())
		if l.skillLoader != nil {
			roundTools = append(roundTools, useSkillToolDef())
		}
		if l.sessionQuerier != nil {
			roundTools = append(roundTools, querySessionToolDef())
		}
		if l.debugLog != nil {
			logLLMCall(l.debugLog, round, msgs, roundTools)
		}
		ch := l.provider.ChatWithTools(ctx, msgs, roundTools, opts)

		// Drain — ChatWithTools emits one final chunk.
		var final llm.StreamChunk
		for chunk := range ch {
			final = chunk
		}

		sp.stop()

		if final.Error != nil {
			return fmt.Errorf("llm: %w", final.Error)
		}

		// Strip a leading <think>...</think> reasoning block some models
		// (Qwen3 and similar) emit inline before their real answer. This is
		// provider-agnostic on purpose: providers aren't guaranteed to split
		// reasoning out on their own (e.g. Ollama's Thinking field requires
		// opting in via ChatRequest.Think, which nothing here sets), and
		// leaving it in risks bloating every subsequent round's history with
		// repeated chain-of-thought noise instead of a clean signal of what
		// was tried and what happened.
		final.Text = stripThinking(final.Text)

		// Belt-and-suspenders: if the provider returned no structured ToolCalls
		// (e.g. native mode is on but the model used a text-format call anyway),
		// try to parse them from the text response.
		if len(final.ToolCalls) == 0 && final.Text != "" {
			if parsed, remainder := llm.ParseTextToolCalls(final.Text); len(parsed) > 0 {
				final.ToolCalls = parsed
				final.Text = remainder
			}
		}

		// ── Tool calls ────────────────────────────────────────────────────
		if len(final.ToolCalls) > 0 {
			l.history = append(l.history, llm.ChatMessage{
				Role:      "assistant",
				Content:   final.Text,
				ToolCalls: final.ToolCalls,
			})

			for _, tc := range final.ToolCalls {
				sig := callSig{tc.Name, fmt.Sprint(tc.Args)}
				recentCalls[sig]++
				if recentCalls[sig] >= 3 {
					l.history = append(l.history, llm.ChatMessage{
						Role:    "user",
						Content: "You appear to be repeating the same tool call. Please give your best answer based on what you have so far.",
					})
					break
				}

				if l.OnToolCall != nil {
					l.OnToolCall(tc.Name, tc.Args)
				}
				if l.debugLog != nil {
					logToolCall(l.debugLog, tc.Name, tc.Args)
				}

				result := l.executeTool(ctx, tc)

				if l.OnToolResult != nil {
					l.OnToolResult(tc.Name, result)
				}
				if l.debugLog != nil {
					logToolResult(l.debugLog, tc.Name, result)
				}

				l.history = append(l.history, llm.ChatMessage{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
					ToolName:   tc.Name,
				})
			}

			sp = l.newSpinnerOrNop()
			continue
		}

		// ── Text response ─────────────────────────────────────────────────
		if final.Text != "" {
			if l.debugLog != nil {
				logResponse(l.debugLog, final.Text)
			}
			onToken(final.Text)
			l.history = append(l.history, llm.ChatMessage{
				Role:    "assistant",
				Content: final.Text,
			})
			return nil
		}

		// Model returned neither tool calls nor text. Nudge it rather than
		// silently returning nothing. After two nudges, give up with an error.
		if nudges >= 2 {
			sp.stop()
			return fmt.Errorf("model returned empty response after %d nudge(s); context may be too degraded", nudges)
		}
		nudges++
		l.history = append(l.history, llm.ChatMessage{
			Role:    "user",
			Content: "Please provide your final answer based on what you have gathered so far.",
		})
		sp = l.newSpinnerOrNop()
	}

	sp.stop()

	// Force a partial response so the user gets something rather than a bare error.
	l.history = append(l.history, llm.ChatMessage{
		Role:    "user",
		Content: "You've reached the tool-call limit for this turn. Based on the work done so far, give your best answer now.",
	})
	msgs := l.buildMessages()
	ch := l.provider.ChatMessages(ctx, msgs, opts)
	var partial strings.Builder
	for chunk := range ch {
		if chunk.Error == nil && chunk.Text != "" {
			partial.WriteString(chunk.Text)
			onToken(chunk.Text)
		}
	}
	if s := partial.String(); s != "" {
		l.history = append(l.history, llm.ChatMessage{Role: "assistant", Content: s})
	}

	return ErrMaxRounds
}

// ClearHistory resets conversation context.
func (l *Loop) ClearHistory() { l.history = l.history[:0] }

// HistoryLen returns the current number of messages in context.
func (l *Loop) HistoryLen() int { return len(l.history) }

// ToolNames returns the names of all tools available to the agent.
func (l *Loop) ToolNames() []string {
	names := make([]string, len(l.tools))
	for i, t := range l.tools {
		names[i] = t.Name
	}
	return names
}

func (l *Loop) buildMessages() []llm.ChatMessage {
	sys := BaseSystemPrompt
	if l.systemPrompt != "" {
		sys = l.systemPrompt
	}

	// Dynamic tool list — external tools plus internal tools.
	internalTools := []llm.ToolDef{taskListToolDef()}
	if l.skillLoader != nil {
		internalTools = append(internalTools, useSkillToolDef())
	}
	if l.sessionQuerier != nil {
		internalTools = append(internalTools, querySessionToolDef())
	}
	allTools := append(l.tools, internalTools...)
	sys += "\n\nAvailable tools:"
	for _, t := range allTools {
		sys += fmt.Sprintf("\n- %s: %s", t.Name, t.Description)
	}

	// Skill catalog — injected after tools so the model knows what skills exist.
	if len(l.skillCatalog) > 0 {
		sys += "\n\nAvailable skills (call use_skill to load full instructions):"
		for _, sk := range l.skillCatalog {
			if !sk.DisableModelInvocation {
				sys += fmt.Sprintf("\n- %s: %s", sk.Name, sk.Description)
			}
		}
	}

	// Queryable sessions — injected when sessions have been merged.
	if len(l.queryableSessions) > 0 {
		sys += "\n\nMerged sessions (queryable via query_session):"
		for _, name := range l.queryableSessions {
			sys += fmt.Sprintf("\n- %s", name)
		}
	}

	// Shell state.
	if c := l.shellCtx; c.CWD != "" {
		sys += fmt.Sprintf("\n\nShell state:\n  cwd: %s", c.CWD)
		if c.LastCommand != "" {
			sys += fmt.Sprintf("\n  last command: %s", c.LastCommand)
			sys += fmt.Sprintf("\n  exit code: %d", c.LastExitCode)
			if c.LastExitCode != 0 && c.LastStderr != "" {
				truncated := c.LastStderr
				if len(truncated) > 500 {
					truncated = truncated[:500] + "..."
				}
				sys += fmt.Sprintf("\n  stderr: %s", truncated)
			}
		}
	}

	// Goal pinning: repeat the current turn's user message in the system prompt so
	// the model can't lose track of the task after many tool calls. This places the
	// goal at the beginning of context (here) AND the end (structurally protected
	// tail), exploiting the U-shaped attention curve.
	if l.turnStart > 0 && l.turnStart < len(l.history) {
		if goal := l.history[l.turnStart]; goal.Role == "user" {
			content := goal.Content
			if len(content) > 300 {
				content = content[:300] + "..."
			}
			sys += fmt.Sprintf("\n\nCurrent task: %s", content)
		}
	}

	// Task list (if the model has populated it). Lives outside history — never compacted.
	if len(l.taskList) > 0 {
		sys += "\n\nTask list:\n" + l.formatTaskList()
	}

	// Active context slots — small slots injected verbatim, large slots as stubs.
	if len(l.contextSlots) > 0 {
		sys += "\n\nActive context:"
		for name, slot := range l.contextSlots {
			if slot.isStub() {
				sys += fmt.Sprintf("\n--- %s [%s, stub] ---\n%s",
					name, humanSize(len(slot.Content)), slot.Description)
			} else {
				sys += fmt.Sprintf("\n--- %s (%s) ---\n%s",
					name, humanSize(len(slot.Content)), slot.Content)
			}
		}
	}

	// Phase 1 — strip verbose tool outputs older than ToolOutputKeepRounds.
	hist := l.stripOldToolOutputs(l.history)

	// Phase 2a — message-count window (secondary guard).
	if max := l.cfg.MaxHistoryMessages; max > 0 && len(hist) > max {
		hist = hist[len(hist)-max:]
		for len(hist) > 0 && hist[0].Role != "user" {
			hist = hist[1:]
		}
	}

	// Phase 2b — token-budget window (primary guard): leave 25% headroom for system prompt.
	if max := l.cfg.MaxContextTokens; max > 0 {
		budget := max * 3 / 4
		for len(hist) > 1 && estimateTokens(hist) > budget {
			hist = hist[1:]
			for len(hist) > 0 && hist[0].Role != "user" {
				hist = hist[1:]
			}
		}
	}

	msgs := make([]llm.ChatMessage, 0, 1+len(hist))
	msgs = append(msgs, llm.ChatMessage{Role: "system", Content: sys})
	return append(msgs, hist...)
}

// humanSize formats a byte count as a human-readable string.
func humanSize(n int) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%dB", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(n)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(n)/(1024*1024))
	}
}

// estimateTokens returns a rough token count for a message slice.
// Uses 4 chars ≈ 1 token, which is slightly conservative for prose and
// roughly correct for code. Accurate enough for budget comparisons.
func estimateTokens(msgs []llm.ChatMessage) int {
	total := 0
	for _, m := range msgs {
		total += len(m.Content) / 4
		for _, tc := range m.ToolCalls {
			total += (len(tc.Name) + len(fmt.Sprint(tc.Args))) / 4
		}
	}
	return total
}

// stripOldToolOutputs replaces the Content of verbose tool result messages
// that are older than ToolOutputKeepRounds rounds with a short placeholder.
// Tool-call/result pairs remain structurally intact — only the content is
// replaced, so the LLM can still follow the exchange sequence.
func (l *Loop) stripOldToolOutputs(hist []llm.ChatMessage) []llm.ChatMessage {
	keep := l.cfg.ToolOutputKeepRounds
	if keep <= 0 || len(hist) == 0 {
		return hist
	}

	// Walk backward counting assistant messages (each = one round) to find
	// the start of the protected tail.
	rounds := 0
	tailStart := len(hist)
	for i := len(hist) - 1; i >= 0; i-- {
		if hist[i].Role == "assistant" {
			rounds++
			if rounds >= keep {
				tailStart = i
				break
			}
		}
	}
	if tailStart == 0 {
		return hist // everything is in the tail
	}

	out := make([]llm.ChatMessage, len(hist))
	copy(out, hist)
	for i := 0; i < tailStart; i++ {
		if out[i].Role == "tool" && len(out[i].Content) > 200 {
			out[i].Content = fmt.Sprintf("[%s result: %d chars — stripped from history]",
				out[i].ToolName, len(out[i].Content))
		}
	}
	return out
}

// shouldCompact reports whether the history has grown past the compaction threshold.
func (l *Loop) shouldCompact() bool {
	if l.cfg.MaxContextTokens <= 0 || l.cfg.CompactionThreshold <= 0 {
		return false
	}
	threshold := float64(l.cfg.MaxContextTokens) * l.cfg.CompactionThreshold
	return float64(estimateTokens(l.history)) >= threshold
}

func (l *Loop) executeTool(ctx context.Context, tc llm.ToolCall) string {
	// Internal tools are handled directly against loop state.
	if tc.Name == "task_list" {
		return l.handleTaskList(tc.Args)
	}
	if tc.Name == "use_skill" {
		return l.handleUseSkill(tc.Args)
	}
	if tc.Name == "query_session" {
		return l.handleQuerySession(ctx, tc.Args)
	}

	handler, ok := l.handlers[tc.Name]
	if !ok {
		return fmt.Sprintf("error: unknown tool %q", tc.Name)
	}
	result, err := handler(tc.Args)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	if strings.TrimSpace(result) == "" {
		return "(no output)"
	}

	// Handle large results.
	if l.cfg.ToolOutputMaxChars > 0 && len(result) > l.cfg.ToolOutputMaxChars {
		if l.cfg.ToolOverflow == "summarize" {
			if summarized, err := l.summarizeLarge(ctx, tc.Name, result); err == nil {
				return summarized
			}
		}
		// Default: truncate.
		return result[:l.cfg.ToolOutputMaxChars] +
			fmt.Sprintf("\n... [truncated — %d chars total]", len(result))
	}

	return result
}

// summarizeLarge calls the LLM to compress a large tool result before storing
// it in history. Falls back to truncation on error.
func (l *Loop) summarizeLarge(ctx context.Context, toolName, content string) (string, error) {
	cap := 20000
	if len(content) < cap {
		cap = len(content)
	}
	prompt := fmt.Sprintf(
		"Summarize the following output from the '%s' tool in 3-5 sentences, preserving key facts, numbers, errors, and file paths:\n\n%s",
		toolName, content[:cap],
	)
	opts := llm.DefaultOptions().WithMaxTokens(300).WithTemperature(0.2)
	ch := l.provider.Ainvoke(ctx, prompt, opts)
	resp, err := llm.CollectResponse(ctx, ch)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("[summarized from %d chars]\n%s", len(content), resp.Text), nil
}

// ── Spinner ───────────────────────────────────────────────────────────────────

type spinnerIface interface{ stop() }

type noopSpinner struct{}

func (s *noopSpinner) stop() {}

type spinner struct {
	stopCh chan struct{}
	doneCh chan struct{}
}

func newSpinner() *spinner {
	s := &spinner{
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}
	go s.run()
	return s
}

func (s *spinner) run() {
	defer close(s.doneCh)
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	tick := time.NewTicker(80 * time.Millisecond)
	defer tick.Stop()
	i := 0
	for {
		select {
		case <-s.stopCh:
			fmt.Print("\r\033[K")
			return
		case <-tick.C:
			fmt.Printf("\r\033[2m%s\033[0m", frames[i%len(frames)])
			i++
		}
	}
}

func (s *spinner) stop() {
	select {
	case <-s.stopCh:
	default:
		close(s.stopCh)
		<-s.doneCh
	}
}
