package agent

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

const debugSep = "══════════════════════════════════════════════════════════════════\n"
const debugLine = "──────────────────────────────────────────────────────────────────\n"

func logLLMCall(w io.Writer, round int, msgs []llm.ChatMessage, tools []llm.ToolDef) {
	fmt.Fprintf(w, "\n%s", debugSep)
	fmt.Fprintf(w, "[%s] LLM CALL  round=%d  messages=%d  tools=%d\n",
		time.Now().Format("15:04:05"), round, len(msgs), len(tools))

	// System prompt (always first message).
	if len(msgs) > 0 && msgs[0].Role == "system" {
		fmt.Fprintf(w, "%s── SYSTEM PROMPT ─────────────────────────────────────────────────\n", debugLine)
		fmt.Fprintln(w, msgs[0].Content)
	}

	// Available tools.
	if len(tools) > 0 {
		fmt.Fprintf(w, "%s── AVAILABLE TOOLS ───────────────────────────────────────────────\n", debugLine)
		for _, t := range tools {
			fmt.Fprintf(w, "  %-24s %s\n", t.Name, t.Description)
		}
	}

	// Message history (skip system prompt already shown).
	history := msgs[1:]
	if len(history) > 0 {
		fmt.Fprintf(w, "%s── MESSAGE HISTORY (%d) ───────────────────────────────────────────\n", debugLine, len(history))
		for _, m := range history {
			switch m.Role {
			case "user":
				fmt.Fprintf(w, "[user] %s\n", truncate(m.Content, 400))
			case "assistant":
				if m.Content != "" {
					fmt.Fprintf(w, "[assistant] %s\n", truncate(m.Content, 400))
				}
				for _, tc := range m.ToolCalls {
					fmt.Fprintf(w, "[assistant→tool] %s(%v)\n", tc.Name, tc.Args)
				}
			case "tool":
				fmt.Fprintf(w, "[tool:%s] %s\n", m.ToolName, truncate(m.Content, 300))
			}
		}
	}

	fmt.Fprintln(w)
}

func logToolCall(w io.Writer, name string, args map[string]interface{}) {
	fmt.Fprintf(w, "  → TOOL CALL: %s(%v)\n", name, args)
}

func logToolResult(w io.Writer, name, result string) {
	fmt.Fprintf(w, "  ← TOOL RESULT [%s]: %s\n\n", name, truncate(result, 500))
}

func logResponse(w io.Writer, text string) {
	fmt.Fprintf(w, "%s── RESPONSE ──────────────────────────────────────────────────────\n", debugLine)
	fmt.Fprintln(w, text)
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + fmt.Sprintf(" … [%d more chars]", len(s)-n)
}
