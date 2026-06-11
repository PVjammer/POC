// Package agent implements a clean multi-turn agent loop using native tool calling.
//
// Design:
//   - Uses ChatWithTools (non-streaming) for all rounds so tool call args are complete
//   - Proper message sequence: system + history, no single-message packing
//   - Tool calls: append assistant msg (with calls) + tool result msgs, then loop
//   - Text response: stream via onToken callback, append to history, done
//   - Loop-breaker: max rounds + repeated-call detection
package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

const (
	maxRounds = 15

	systemPrompt = `You are an AI shell assistant running inside a Unix terminal.
You help users accomplish tasks using shell commands and your own knowledge.
You have a bash tool. Use it to execute commands, read files, check system state, etc.
Be concise. When you run commands, show the relevant output. Prefer doing over explaining.
If the user asks a question you can answer directly, do so without using bash.`
)

// Loop is a stateful, multi-turn agent. One Loop instance persists across
// multiple user turns so conversation history is maintained.
type Loop struct {
	provider llm.ToolCallingProvider
	tools    []llm.ToolDef
	handlers map[string]func(map[string]interface{}) (string, error)
	history  []llm.ChatMessage

	// onToolCall is called before each tool execution so the shell can print
	// a status line. Receives the tool name and args.
	OnToolCall func(name string, args map[string]interface{})
	// onToolResult is called after each tool execution with the result.
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

// Run executes one user turn (possibly many agent rounds).
// onToken is called with each text token of the final response.
func (l *Loop) Run(ctx context.Context, userMsg string, onToken func(string)) error {
	l.history = append(l.history, llm.ChatMessage{
		Role:    "user",
		Content: userMsg,
	})

	opts := &llm.CompletionOptions{
		MaxTokens:   4096,
		Temperature: 0.3, // lower = more reliable tool use
	}

	// Track recent tool calls to detect infinite loops.
	type callSig struct{ name, args string }
	recentCalls := make(map[callSig]int)

	for round := 0; round < maxRounds; round++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		msgs := l.buildMessages()
		ch := l.provider.ChatWithTools(ctx, msgs, l.tools, opts)

		// Drain the channel — ChatWithTools sends one final chunk.
		var final llm.StreamChunk
		for chunk := range ch {
			final = chunk
		}

		if final.Error != nil {
			return fmt.Errorf("llm: %w", final.Error)
		}

		// ── Tool calls ────────────────────────────────────────────────────
		if len(final.ToolCalls) > 0 {
			// Append the assistant turn that requested the calls.
			l.history = append(l.history, llm.ChatMessage{
				Role:      "assistant",
				Content:   final.Text, // may be empty, that's fine
				ToolCalls: final.ToolCalls,
			})

			for _, tc := range final.ToolCalls {
				// Loop-breaker: same tool + args called 3 times → bail.
				sig := callSig{tc.Name, fmt.Sprint(tc.Args)}
				recentCalls[sig]++
				if recentCalls[sig] >= 3 {
					l.history = append(l.history, llm.ChatMessage{
						Role:    "user",
						Content: "You appear to be stuck in a loop. Please give a final answer based on what you know so far.",
					})
					break
				}

				result := l.executeTool(tc)

				if l.OnToolCall != nil {
					l.OnToolCall(tc.Name, tc.Args)
				}
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
			continue // next round
		}

		// ── Text response ─────────────────────────────────────────────────
		if final.Text != "" {
			// Deliver full text via the callback.
			// TODO: add streaming path when ChatWithTools supports it.
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
func (l *Loop) ClearHistory() {
	l.history = l.history[:0]
}

// HistoryLen returns the number of messages in the current session.
func (l *Loop) HistoryLen() int { return len(l.history) }

func (l *Loop) buildMessages() []llm.ChatMessage {
	msgs := make([]llm.ChatMessage, 0, 1+len(l.history))
	msgs = append(msgs, llm.ChatMessage{
		Role:    "system",
		Content: systemPrompt,
	})
	return append(msgs, l.history...)
}

func (l *Loop) executeTool(tc llm.ToolCall) string {
	handler, ok := l.handlers[tc.Name]
	if !ok {
		return fmt.Sprintf("error: unknown tool %q", tc.Name)
	}
	result, err := handler(tc.Args)
	if err != nil {
		return fmt.Sprintf("error executing %s: %v", tc.Name, err)
	}
	if strings.TrimSpace(result) == "" {
		return "(no output)"
	}
	return result
}
