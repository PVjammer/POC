package agent

import (
	"fmt"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

func useSkillToolDef() llm.ToolDef {
	return llm.ToolDef{
		Name:        "use_skill",
		Description: "Load a skill's full instructions by name. Call this when a skill listed in the available skills catalog is relevant to the current task. The skill body will be returned and should guide your next actions.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "The skill name from the available skills catalog",
				},
			},
			"required": []string{"name"},
		},
	}
}

func (l *Loop) handleUseSkill(args map[string]interface{}) string {
	name, _ := args["name"].(string)
	if name == "" {
		return "error: name is required"
	}

	// Check catalog — respect DisableModelInvocation.
	for _, sk := range l.skillCatalog {
		if sk.Name == name {
			if sk.DisableModelInvocation {
				return fmt.Sprintf("error: skill %q cannot be invoked by the model (user-only)", name)
			}
			break
		}
	}

	if l.skillLoader == nil {
		return "error: skill loader not available"
	}

	body, err := l.skillLoader.Body(name)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	if body == "" {
		return fmt.Sprintf("skill %q has no body content", name)
	}
	return body
}
