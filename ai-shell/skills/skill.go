// Package skills implements agentskills.io-compatible skill discovery and loading.
// A skill is a directory containing a SKILL.md file with YAML frontmatter.
package skills

// Record holds the parsed metadata from a skill's SKILL.md frontmatter.
type Record struct {
	Name                   string   // must match directory name
	Description            string
	ArgumentHint           string   // shown in /skill list and tab completion
	AllowedTools           []string // parsed but not enforced; permissions TBD
	DisableModelInvocation bool     // true = agent cannot activate; user /command only
	Dir                    string   // absolute path to the skill directory
}

// SkillLoader is the interface agent/loop.go uses to avoid a circular import.
// The concrete *Loader satisfies this interface.
type SkillLoader interface {
	Get(name string) (Record, bool)
	Body(name string) (string, error)
}
