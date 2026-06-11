package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

const (
	bashTimeout  = 30 * time.Second
	maxOutputLen = 10_000
)

// BashToolDef returns the tool schema the LLM sees.
func BashToolDef() llm.ToolDef {
	return llm.ToolDef{
		Name:        "bash",
		Description: "Execute a bash shell command and return combined stdout+stderr. Use for file operations, searching, running scripts, checking system state, etc.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"command": map[string]interface{}{
					"type":        "string",
					"description": "The bash command to execute",
				},
			},
			"required": []string{"command"},
		},
	}
}

// BashHandler executes a command and captures its output (no PTY — for agent use).
// Interactive programs like vim cannot be run this way; use the shell's ! prefix for those.
func BashHandler(args map[string]interface{}) (string, error) {
	command, ok := args["command"].(string)
	if !ok || strings.TrimSpace(command) == "" {
		return "", fmt.Errorf("command argument is required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), bashTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()

	var sb strings.Builder
	if stdout.Len() > 0 {
		sb.WriteString(stdout.String())
	}
	if stderr.Len() > 0 {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("[stderr] ")
		sb.WriteString(stderr.String())
	}

	out := strings.TrimRight(sb.String(), "\n")

	if out == "" {
		if runErr != nil {
			return fmt.Sprintf("command failed (exit %v)", runErr), nil
		}
		return "(no output)", nil
	}

	if len(out) > maxOutputLen {
		out = out[:maxOutputLen] + fmt.Sprintf("\n... (truncated — %d chars total)", len(out))
	}

	return out, nil
}

// AllTools returns the tool definitions to pass to the agent.
func AllTools() []llm.ToolDef {
	return []llm.ToolDef{BashToolDef()}
}

// AllHandlers returns the tool name → execution function map.
func AllHandlers() map[string]func(map[string]interface{}) (string, error) {
	return map[string]func(map[string]interface{}) (string, error){
		"bash": BashHandler,
	}
}
