package shell

import "strings"

// InputType classifies user input for routing.
type InputType int

const (
	// InputDirect: plain text or !cmd — pass to bash
	InputDirect InputType = iota
	// InputAgent: ?msg — advisory AI (no tool execution)
	InputAgent
	// InputAgentAct: !"msg" — agentic AI (may execute commands; permissions apply)
	InputAgentAct
	// InputMeta: /cmd or /fn — built-in command or AI function
	InputMeta
	// InputPipeline: <bash_cmd> | /fn  or  <bash_cmd> | ?msg
	// The left side is run as bash; its stdout becomes stdin for the right side.
	InputPipeline
	// InputBash: a plain bash command appearing on the right side of a pipeline
	// from an AI command. e.g. /job 1 | grep foo — "grep foo" is InputBash.
	InputBash
)

// ParsedInput is the result of classifying a line of user input.
type ParsedInput struct {
	Type    InputType
	Content string // stripped of routing prefix (Direct/Agent/Meta)
	// Pipeline fields — only set when Type == InputPipeline.
	PipeLeft  string       // bash command to run
	PipeRight *ParsedInput // /fn or ?msg to receive its output
}

// Parse classifies raw input and strips routing prefixes.
// Empty input returns a zero ParsedInput (Content == "").
func Parse(raw string) ParsedInput {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ParsedInput{}
	}

	// Detect "bash_cmd | /fn" or "bash_cmd | ?msg" pipelines.
	// Pure bash pipelines (cat file | grep foo) are left unchanged and
	// go to bash directly — only the last segment triggers this.
	if idx := pipeToAI(raw); idx >= 0 {
		left := strings.TrimSpace(raw[:idx])
		right := Parse(strings.TrimSpace(raw[idx+1:]))
		return ParsedInput{
			Type:      InputPipeline,
			PipeLeft:  left,
			PipeRight: &right,
		}
	}

	// Detect "/ai-cmd | bash_cmd" — AI command on the left piping into plain bash.
	// e.g. "/job 1 | grep foo", "/ctx show arch | wc -l"
	if idx := pipeFromAI(raw); idx >= 0 {
		left := strings.TrimSpace(raw[:idx])
		right := strings.TrimSpace(raw[idx+1:])
		bashRight := ParsedInput{Type: InputBash, Content: right}
		return ParsedInput{
			Type:      InputPipeline,
			PipeLeft:  left,
			PipeRight: &bashRight,
		}
	}

	switch {
	case strings.HasPrefix(raw, "!"):
		rest := raw[1:]
		// !"..." or !'...' → agentic mode; bare !cmd → direct bash (history expansion etc.)
		if len(rest) > 0 && (rest[0] == '"' || rest[0] == '\'') {
			return ParsedInput{Type: InputAgentAct, Content: strings.TrimSpace(stripOuterQuotes(rest))}
		}
		return ParsedInput{Type: InputDirect, Content: strings.TrimSpace(rest)}
	case strings.HasPrefix(raw, "?"):
		rest := raw[1:]
		// Optional quotes — ?"..." and ?... are both advisory
		if len(rest) > 0 && (rest[0] == '"' || rest[0] == '\'') {
			return ParsedInput{Type: InputAgent, Content: strings.TrimSpace(stripOuterQuotes(rest))}
		}
		return ParsedInput{Type: InputAgent, Content: strings.TrimSpace(rest)}
	case strings.HasPrefix(raw, "/"):
		return ParsedInput{Type: InputMeta, Content: strings.TrimSpace(raw[1:])}
	default:
		// Plain text → bash, just like a real shell.
		return ParsedInput{Type: InputDirect, Content: raw}
	}
}

// pipeToAI scans left-to-right for the first `|` whose right-hand side is an
// AI segment (? query or /function). Scanning left-to-right means chained AI
// pipes like "cat f | /summarize | /ctx add" split at the first AI boundary,
// letting the recursive parser build the full chain correctly.
//
// An AI segment starting with / must be a bare name (no path separator in the
// first token) to distinguish "/summarize" from "/usr/bin/grep".
func pipeToAI(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == '|' {
			right := strings.TrimSpace(s[i+1:])
			if isAISegment(right) {
				return i
			}
		}
	}
	return -1
}

// pipeFromAI scans left-to-right for the first `|` whose LEFT-hand side is an
// AI segment and right-hand side is plain bash. This handles "/job 1 | grep foo"
// and "/ctx show arch | wc -l". Called only after pipeToAI returns -1.
func pipeFromAI(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == '|' {
			left := strings.TrimSpace(s[:i])
			right := strings.TrimSpace(s[i+1:])
			if isAISegment(left) && !isAISegment(right) && right != "" {
				return i
			}
		}
	}
	return -1
}

// isAISegment reports whether a trimmed string looks like an AI routing prefix.
func isAISegment(s string) bool {
	if strings.HasPrefix(s, "?") {
		return true
	}
	// !"..." is an agentic segment
	if len(s) > 1 && s[0] == '!' && (s[1] == '"' || s[1] == '\'') {
		return true
	}
	if !strings.HasPrefix(s, "/") {
		return false
	}
	// Must be a bare name — no path separator in the first token.
	// This ensures "/summarize" matches but "/usr/bin/grep foo" does not.
	first := strings.Fields(s[1:])
	return len(first) > 0 && !strings.Contains(first[0], "/")
}

// stripOuterQuotes removes a matching pair of surrounding " or ' characters.
// If the string has an opening quote but no matching closing quote, only the
// opening quote is removed. Non-quoted strings are returned unchanged.
func stripOuterQuotes(s string) string {
	if len(s) == 0 {
		return s
	}
	q := s[0]
	if q != '"' && q != '\'' {
		return s
	}
	if len(s) >= 2 && s[len(s)-1] == q {
		return s[1 : len(s)-1]
	}
	return s[1:] // unclosed quote — strip opening only
}
