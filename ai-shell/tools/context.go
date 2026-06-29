package tools

import "github.com/pvjammer/ai-sdk-go/pkg/llm"

// ReadContextToolDef returns the schema for the read_context tool.
// The handler is a closure created in shell.New() that captures ctxSlots.
func ReadContextToolDef() llm.ToolDef {
	return llm.ToolDef{
		Name:        "read_context",
		Description: "Read the content of a named context slot. Large slots are returned in pages; use offset to continue reading. Use query to search for specific content instead of reading linearly.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "Name of the context slot to read",
				},
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Return only lines matching these search terms (with surrounding context). Prefer this over reading linearly when looking for specific information.",
				},
				"offset": map[string]interface{}{
					"type":        "integer",
					"description": "Byte offset to start reading from. Use the value from the previous call's continuation footer to page through large slots.",
				},
			},
			"required": []string{"name"},
		},
	}
}

// DescribeToolToolDef returns the schema for the describe_tool tool.
// The handler is a closure created in shell.New() that captures the agent loop.
func DescribeToolToolDef() llm.ToolDef {
	return llm.ToolDef{
		Name:        "describe_tool",
		Description: "Get the full JSON schema for any available tool by name.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "Name of the tool to describe",
				},
			},
			"required": []string{"name"},
		},
	}
}
