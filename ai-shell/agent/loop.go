// Package agent implements a clean multi-turn agent loop using native tool calling.
//
// Design:
//   - Uses ChatWithTools (non-streaming) for all rounds; tool call args must be complete
//   - Proper message sequence: system + history, no single-message packing
//   - Spinner during LLM wait, tool call display during execution
//   - Shell context (cwd, last cmd, exit code) injected into system prompt each turn
//   - Loop-breaker: max rounds + repeated-identical-call detection
package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

const (
	maxRounds = 15

	baseSystemPrompt = `You are an AI shell assistant running inside a Unix terminal.
You help users accomplish tasks using shell commands and your own knowledge.
You have a bash tool and other utility tools. Use them to get things done.
Be concise. Show relevant command output. Prefer doing over explaining.
If the user asks something you can answer directly, do so without calling tools.`
)

// ShellContext holds the current state of the shell, injected into the agent
// system prompt so the AI has accurate context for each turn.
type ShellContext struct {
	CWD          string
	LastCommand  string
	LastExitCode int
	LastStderr   string // captured stderr on non-zero exit, for "?why" style queries
}

// Loop is a stateful, multi-turn agent. One instance persists across user turns
// so conversation history is maintained for the session.
type Loop struct {
	provider llm.ToolCallingProvider
	tools    []llm.ToolDef
	handlers map[string]func(map[string]interface{}) (string, error)
	history  []llm.ChatMessage
	shellCtx ShellContext

	// Callbacks for the shell to display tool activity.
	OnToolCall   func(name string, args map[string]interface{})
	OnToolResult func(name string, result string)
}

// New creates a new agent loop.
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
	}
}

// SetShellContext updates the shell state that is injected into every LLM call.
// Call this before Run() each turn so the agent always has current state.
func (l *Loop) SetShellContext(ctx ShellContext) {
	l.shellCtx = ctx
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
					// Stuck in a loop — force a final answer.
					l.history = append(l.history, llm.ChatMessage{
						Role:    "user",
						Content: "You appear to be repeating the same tool call. Please give your best answer based on what you have so far.",
					})
					break
				}

				if l.OnToolCall != nil {
					l.OnToolCall(tc.Name, tc.Args)
				}

				result := l.executeTool(tc)

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

			sp = newSpinner() // spinner for next round
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

	msgs := make([]llm.ChatMessage, 0, 1+len(l.history))
	msgs = append(msgs, llm.ChatMessage{Role: "system", Content: sys})
	return append(msgs, l.history...)
}

func (l *Loop) executeTool(tc llm.ToolCall) string {
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
	return result
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
			fmt.Print("\r\033[K") // clear spinner line
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
		// already stopped
	default:
		close(s.stopCh)
		<-s.doneCh
	}
}
