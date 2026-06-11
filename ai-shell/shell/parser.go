package shell

import "strings"

// InputType classifies user input for routing.
type InputType int

const (
	// InputDirect: plain text or !cmd — pass to bash
	InputDirect InputType = iota
	// InputAgent: ?msg — send to AI agent
	InputAgent
	// InputMeta: /cmd or /fn — built-in command or AI function
	InputMeta
	// InputPipeline: <bash_cmd> | /fn  or  <bash_cmd> | ?msg
	// The left side is run as bash; its stdout becomes stdin for the right side.
	InputPipeline
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

	switch {
	case strings.HasPrefix(raw, "!"):
		// Explicit bash override (redundant now, kept for muscle memory).
		return ParsedInput{Type: InputDirect, Content: strings.TrimSpace(raw[1:])}
	case strings.HasPrefix(raw, "?"):
		return ParsedInput{Type: InputAgent, Content: strings.TrimSpace(raw[1:])}
	case strings.HasPrefix(raw, "/"):
		return ParsedInput{Type: InputMeta, Content: strings.TrimSpace(raw[1:])}
	default:
		// Plain text → bash, just like a real shell.
		return ParsedInput{Type: InputDirect, Content: raw}
	}
}

// pipeToAI scans right-to-left for the last `|` whose right-hand side
// starts with / or ? (an AI-side segment). Returns -1 if not found.
func pipeToAI(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '|' {
			right := strings.TrimSpace(s[i+1:])
			if strings.HasPrefix(right, "/") || strings.HasPrefix(right, "?") {
				return i
			}
		}
	}
	return -1
}
