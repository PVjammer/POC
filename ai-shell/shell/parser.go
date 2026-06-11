package shell

import "strings"

// InputType classifies user input for routing.
type InputType int

const (
	// InputDirect: !cmd — pass straight to bash (PTY passthrough, vim works)
	InputDirect InputType = iota
	// InputAgent: ?msg or plain text — send to AI agent
	InputAgent
	// InputMeta: /cmd — built-in shell commands
	InputMeta
)

// ParsedInput is the result of classifying a line of user input.
type ParsedInput struct {
	Type    InputType
	Content string // stripped of the routing prefix
}

// Parse classifies a raw input line and strips its prefix.
// Empty input returns a zero ParsedInput (Content == "").
func Parse(raw string) ParsedInput {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ParsedInput{}
	}

	switch {
	case strings.HasPrefix(raw, "!"):
		return ParsedInput{Type: InputDirect, Content: strings.TrimSpace(raw[1:])}
	case strings.HasPrefix(raw, "?"):
		return ParsedInput{Type: InputAgent, Content: strings.TrimSpace(raw[1:])}
	case strings.HasPrefix(raw, "/"):
		return ParsedInput{Type: InputMeta, Content: strings.TrimSpace(raw[1:])}
	default:
		// Plain text → bash, just like a real shell.
		// Use ? to talk to the AI.
		return ParsedInput{Type: InputDirect, Content: raw}
	}
}
