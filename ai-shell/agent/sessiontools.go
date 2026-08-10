package agent

import (
	"context"
	"fmt"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

func querySessionToolDef() llm.ToolDef {
	return llm.ToolDef{
		Name:        "query_session",
		Description: "Ask a question to a merged session and get an answer based on that session's full conversation history. Use this when you need specific details from a session that was merged into this one.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"session": map[string]interface{}{
					"type":        "string",
					"description": "Name of the merged session to query",
				},
				"question": map[string]interface{}{
					"type":        "string",
					"description": "Question to ask the session, e.g. \"what approach did you use for X?\"",
				},
			},
			"required": []string{"session", "question"},
		},
	}
}

func (l *Loop) handleQuerySession(ctx context.Context, args map[string]interface{}) string {
	if l.sessionQuerier == nil {
		return "error: no queryable sessions available"
	}
	name, _ := args["session"].(string)
	question, _ := args["question"].(string)
	if name == "" || question == "" {
		return "error: query_session requires 'session' and 'question' arguments"
	}
	result, err := l.sessionQuerier.QuerySession(ctx, name, question)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	return result
}
