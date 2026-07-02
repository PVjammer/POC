package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// runCompaction summarizes the middle portion of conversation history with an
// LLM call, replacing it with a single structured summary message. The head
// (first exchange) and tail (last CompactionTailMessages messages) are always
// kept verbatim. This is non-destructive: a compaction failure leaves
// l.history unchanged so the caller can continue normally.
func (l *Loop) runCompaction(ctx context.Context) error {
	hist := l.history

	headEnd := findHeadEnd(hist)
	tailStart := findTailStart(hist, l.cfg.CompactionTailMessages)

	// Nothing useful in the middle to compact.
	if tailStart <= headEnd+2 {
		return nil
	}

	head := hist[:headEnd]
	middle := hist[headEnd:tailStart]
	tail := hist[tailStart:]

	prompt := l.buildCompactionPrompt(middle)

	opts := llm.DefaultOptions().WithMaxTokens(600).WithTemperature(0.1)
	ch := l.provider.Ainvoke(ctx, prompt, opts)
	resp, err := llm.CollectResponse(ctx, ch)
	if err != nil {
		return fmt.Errorf("compaction LLM call: %w", err)
	}
	summary := strings.TrimSpace(resp.Text)
	if summary == "" {
		return fmt.Errorf("compaction returned empty summary")
	}

	summaryMsg := llm.ChatMessage{
		Role:    "assistant",
		Content: "[Compacted — session summary follows]\n\n" + summary,
	}

	rebuilt := make([]llm.ChatMessage, 0, len(head)+1+len(tail))
	rebuilt = append(rebuilt, head...)
	rebuilt = append(rebuilt, summaryMsg)
	rebuilt = append(rebuilt, tail...)
	l.history = rebuilt

	l.compactionSummary = summary
	l.compactionDepth++
	return nil
}

func (l *Loop) buildCompactionPrompt(middle []llm.ChatMessage) string {
	histText := formatHistoryForCompaction(middle)

	if l.compactionSummary == "" {
		return fmt.Sprintf(`Summarize this conversation segment in the following structured format. Be concise; use bullet points within each section. Omit sections that have no content.

**Goal:** What the user is trying to accomplish
**Progress:** What has been completed
**Key Decisions:** Important choices or approaches selected
**Relevant Files:** Files created, modified, or discussed
**Errors & Fixes:** Errors encountered and how they were resolved
**Open Questions:** Unresolved issues or pending decisions
**Next Steps:** What should happen next

Conversation:
%s`, histText)
	}

	return fmt.Sprintf(`Update this session summary to incorporate the new conversation below. Move completed items from "Next Steps" to "Progress". Add new decisions, files, and errors. Remove information that is no longer relevant. Return only the updated summary in the same structured format.

Existing summary:
%s

New conversation to incorporate:
%s`, l.compactionSummary, histText)
}

// findHeadEnd returns the exclusive end index of the first user→assistant exchange.
func findHeadEnd(hist []llm.ChatMessage) int {
	seenUser := false
	for i, m := range hist {
		if m.Role == "user" {
			if seenUser {
				return i
			}
			seenUser = true
		}
	}
	return len(hist)
}

// findTailStart returns the start index of the protected tail.
// It aligns to the nearest user-message boundary.
func findTailStart(hist []llm.ChatMessage, tailMessages int) int {
	if tailMessages <= 0 || tailMessages >= len(hist) {
		return len(hist)
	}
	idx := len(hist) - tailMessages
	if idx < 0 {
		idx = 0
	}
	// Walk forward to the next user message boundary.
	for idx < len(hist) && hist[idx].Role != "user" {
		idx++
	}
	return idx
}

// formatHistoryForCompaction renders a message slice as plain text for the
// compaction LLM prompt.
func formatHistoryForCompaction(msgs []llm.ChatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case "user":
			sb.WriteString("User: ")
			sb.WriteString(m.Content)
		case "assistant":
			if m.Content != "" {
				sb.WriteString("Assistant: ")
				sb.WriteString(m.Content)
			}
			for _, tc := range m.ToolCalls {
				sb.WriteString(fmt.Sprintf("Assistant called tool '%s'", tc.Name))
			}
		case "tool":
			content := m.Content
			if len(content) > 500 {
				content = content[:500] + "..."
			}
			name := m.ToolName
			if name == "" {
				name = "tool"
			}
			sb.WriteString(fmt.Sprintf("Tool '%s' returned: %s", name, content))
		}
		sb.WriteString("\n\n")
	}
	return sb.String()
}
