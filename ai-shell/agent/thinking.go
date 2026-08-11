package agent

import "strings"

// stripThinking removes a leading <think>...</think> reasoning block from s.
// Reasoning models (Qwen3 and similar) emit this ahead of their real answer
// when a server doesn't split it out on its own; left inline it pollutes the
// conversation history every subsequent round is built from. A block that
// opens but never closes — truncated by a token limit — has no recoverable
// answer and is dropped entirely rather than left dangling.
func stripThinking(s string) string {
	const openTag = "<think>"
	const closeTag = "</think>"

	if !strings.HasPrefix(s, openTag) {
		return s
	}
	idx := strings.Index(s, closeTag)
	if idx < 0 {
		return ""
	}
	return strings.TrimSpace(s[idx+len(closeTag):])
}
