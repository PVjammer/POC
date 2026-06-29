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
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

const maxRounds = 15

const baseSystemPrompt = `You are an AI shell assistant running inside a Unix terminal.
Help the user accomplish tasks using shell commands and your own knowledge.
Be concise. Show relevant output. Prefer doing over explaining.
Use tools when needed; answer directly when you can.

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
	systemPrompt   string // empty = use baseSystemPrompt

	// Compaction state (Phase 4).
	compactionSummary string // current structured summary; empty = never compacted
	compactionDepth   int    // number of times compaction has run this session

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
}

// SetTools replaces the tool list (called when functions are reloaded).
func (l *Loop) SetTools(tools []llm.ToolDef, handlers map[string]func(map[string]interface{}) (string, error)) {
	l.tools = tools
	l.handlers = handlers
}

// SetProvider replaces the LLM provider (called when the model is switched mid-session).
func (l *Loop) SetProvider(p llm.ToolCallingProvider) { l.provider = p }

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
	l.history = append(l.history, llm.ChatMessage{
		Role:    "user",
		Content: userMsg,
	})

	opts := &llm.CompletionOptions{
		MaxTokens:   4096,
		Temperature: 0.3,
	}

	type callSig struct{ name, args string }
	recentCalls := make(map[callSig]int)

	sp := l.newSpinnerOrNop()

	for round := 0; round < maxRounds; round++ {
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
		if l.debugLog != nil {
			logLLMCall(l.debugLog, round, msgs, l.tools)
		}
		ch := l.provider.ChatWithTools(ctx, msgs, l.tools, opts)

		// Drain — ChatWithTools emits one final chunk.
		var final llm.StreamChunk
		for chunk := range ch {
			final = chunk
		}

		sp.stop()

		if final.Error != nil {
			return fmt.Errorf("llm: %w", final.Error)
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
		}

		return nil
	}

	sp.stop()
	return fmt.Errorf("reached max rounds (%d) without a final response", maxRounds)
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
	sys := baseSystemPrompt
	if l.systemPrompt != "" {
		sys = l.systemPrompt
	}

	// Dynamic tool list.
	if len(l.tools) > 0 {
		sys += "\n\nAvailable tools:"
		for _, t := range l.tools {
			sys += fmt.Sprintf("\n- %s: %s", t.Name, t.Description)
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
