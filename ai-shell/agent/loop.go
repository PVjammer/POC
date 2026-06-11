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
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

const maxRounds = 15

const baseSystemPrompt = `You are an AI shell assistant running inside a Unix terminal.
Help the user accomplish tasks using shell commands and your own knowledge.
Be concise. Show relevant output. Prefer doing over explaining.
Use tools when needed; answer directly when you can.`

// LoopConfig holds tuneable parameters for the agent loop.
type LoopConfig struct {
	MaxHistoryMessages int    // number of messages to keep in context (default 20)
	ToolOutputMaxChars int    // truncate/summarize tool results above this (default 4000)
	ToolOverflow       string // "truncate" or "summarize"
	CtxMaxInjectChars  int    // max chars to inject per context slot (default 8000)
}

func defaultLoopConfig() LoopConfig {
	return LoopConfig{
		MaxHistoryMessages: 20,
		ToolOutputMaxChars: 4000,
		ToolOverflow:       "truncate",
		CtxMaxInjectChars:  8000,
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
	provider     llm.ToolCallingProvider
	tools        []llm.ToolDef
	handlers     map[string]func(map[string]interface{}) (string, error)
	history      []llm.ChatMessage
	shellCtx     ShellContext
	contextSlots map[string]string
	cfg          LoopConfig

	// Callbacks for the shell to display tool activity.
	OnToolCall   func(name string, args map[string]interface{})
	OnToolResult func(name string, result string)
}

// New creates a new agent loop with default configuration.
func New(
	provider llm.ToolCallingProvider,
	tools []llm.ToolDef,
	handlers map[string]func(map[string]interface{}) (string, error),
) *Loop {
	return &Loop{
		provider: provider,
		tools:    tools,
		handlers: handlers,
		history:  make([]llm.ChatMessage, 0, 32),
		cfg:      defaultLoopConfig(),
	}
}

// SetShellContext updates the shell state injected into every LLM call.
func (l *Loop) SetShellContext(ctx ShellContext) { l.shellCtx = ctx }

// SetContextSlots updates named context content injected into the system prompt.
func (l *Loop) SetContextSlots(slots map[string]string) { l.contextSlots = slots }

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
	if cfg.CtxMaxInjectChars > 0 {
		l.cfg.CtxMaxInjectChars = cfg.CtxMaxInjectChars
	}
}

// SetTools replaces the tool list (called when functions are reloaded).
func (l *Loop) SetTools(tools []llm.ToolDef, handlers map[string]func(map[string]interface{}) (string, error)) {
	l.tools = tools
	l.handlers = handlers
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

	sp := newSpinner()

	for round := 0; round < maxRounds; round++ {
		if ctx.Err() != nil {
			sp.stop()
			return ctx.Err()
		}

		msgs := l.buildMessages()
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

				result := l.executeTool(ctx, tc)

				if l.OnToolResult != nil {
					l.OnToolResult(tc.Name, result)
				}

				l.history = append(l.history, llm.ChatMessage{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
					ToolName:   tc.Name,
				})
			}

			sp = newSpinner()
			continue
		}

		// ── Text response ─────────────────────────────────────────────────
		if final.Text != "" {
			onToken(final.Text)
			l.history = append(l.history, llm.ChatMessage{
				Role:    "assistant",
				Content: final.Text,
			})
		}

		return nil
	}

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

	// Active context slots — capped to CtxMaxInjectChars per slot.
	if len(l.contextSlots) > 0 {
		sys += "\n\nActive context:"
		for name, content := range l.contextSlots {
			cap := l.cfg.CtxMaxInjectChars
			if cap > 0 && len(content) > cap {
				sys += fmt.Sprintf("\n--- %s (%d chars total, showing first %d) ---\n%s\n[... truncated]",
					name, len(content), cap, content[:cap])
			} else {
				sys += fmt.Sprintf("\n--- %s (%d chars) ---\n%s", name, len(content), content)
			}
		}
	}

	// Trim history to configured window, aligning to a user-message boundary.
	hist := l.history
	if max := l.cfg.MaxHistoryMessages; max > 0 && len(hist) > max {
		hist = hist[len(hist)-max:]
		for len(hist) > 0 && hist[0].Role != "user" {
			hist = hist[1:]
		}
	}

	msgs := make([]llm.ChatMessage, 0, 1+len(hist))
	msgs = append(msgs, llm.ChatMessage{Role: "system", Content: sys})
	return append(msgs, hist...)
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
