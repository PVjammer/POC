package agent

import (
	"fmt"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

type taskEntry struct {
	id   int
	text string
	done bool
}

func taskListToolDef() llm.ToolDef {
	return llm.ToolDef{
		Name:        "task_list",
		Description: "Scratchpad task list for the current query. Use for complex multi-step tasks only — skip for simple lookups. Resets with each new query and is never compacted.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"action": map[string]interface{}{
					"type":        "string",
					"enum":        []string{"add", "done", "list"},
					"description": "add: append a new task; done: mark a task complete by id; list: show all tasks",
				},
				"task": map[string]interface{}{
					"type":        "string",
					"description": "Task description (required for add)",
				},
				"id": map[string]interface{}{
					"type":        "integer",
					"description": "Task ID to mark done (required for done)",
				},
			},
			"required": []string{"action"},
		},
	}
}

func (l *Loop) handleTaskList(args map[string]interface{}) string {
	action, _ := args["action"].(string)
	switch action {
	case "add":
		text, _ := args["task"].(string)
		if strings.TrimSpace(text) == "" {
			return "error: task text is required"
		}
		id := l.nextTaskID
		l.nextTaskID++
		l.taskList = append(l.taskList, taskEntry{id: id, text: text})
		return fmt.Sprintf("added task %d", id)

	case "done":
		var id int
		switch v := args["id"].(type) {
		case float64:
			id = int(v)
		case int:
			id = v
		}
		if id == 0 {
			return "error: id is required for done"
		}
		for i, t := range l.taskList {
			if t.id == id {
				l.taskList[i].done = true
				return fmt.Sprintf("task %d done: %s", id, t.text)
			}
		}
		return fmt.Sprintf("error: no task with id %d", id)

	case "list":
		return l.formatTaskList()

	default:
		return "error: action must be one of: add, done, list"
	}
}

func (l *Loop) formatTaskList() string {
	if len(l.taskList) == 0 {
		return "(no tasks)"
	}
	var sb strings.Builder
	for _, t := range l.taskList {
		mark := "[ ]"
		if t.done {
			mark = "[x]"
		}
		fmt.Fprintf(&sb, "%s %d. %s\n", mark, t.id, t.text)
	}
	return strings.TrimRight(sb.String(), "\n")
}
